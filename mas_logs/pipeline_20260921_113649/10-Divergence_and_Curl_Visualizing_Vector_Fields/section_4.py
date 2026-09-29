from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Graphical Synthesis & Comparison", [
            "Compare radial expansion and circular rotation.",
            "Field A shows divergence without curl.",
            "Field B shows curl without divergence."
        ])
        
        # StreamLines are expensive, using a smaller resolution
        def radial_field(p):
            return np.array([p[0], p[1], 0]) * 0.2
        def circular_field(p):
            return np.array([-p[1], p[0], 0]) * 0.2

        field_a = StreamLines(radial_field, stroke_width=2, color=BLUE, x_range=[-1.5, 1.5], y_range=[-1.5, 1.5])
        field_b = StreamLines(circular_field, stroke_width=2, color=RED, x_range=[-1.5, 1.5], y_range=[-1.5, 1.5])
        
        # Load assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        turbine = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/turbine.svg")

        # Prep placeholders
        label_a = Text("Field A", font_size=20, color=BLUE)
        label_b = Text("Field B", font_size=20, color=RED)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(field_a, "B2", scale_factor=0.6)
        self.place_at_grid(field_b, "B5", scale_factor=0.6)
        self.place_at_grid(compass, "A1", scale_factor=0.4)
        self.play(Create(field_a), Create(field_b), FadeIn(compass))
        field_a.start_animation()
        field_b.start_animation()
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        self.place_at_grid(label_a, "B3", scale_factor=0.6)
        self.play(Write(label_a))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        self.place_at_grid(label_b, "B6", scale_factor=0.6)
        self.place_in_area(turbine, "D3", "E4", scale_factor=0.7)
        self.play(Write(label_b), FadeIn(turbine))
        self.wait(2)

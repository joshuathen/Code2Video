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
        self.setup_layout("The Unified Visual Test", ["Inspect fields with two simple tests.", "Balloon expansion detects positive divergence.", "Propeller rotation reveals non-zero curl."])
        
        # Define basic flow components
        field = ArrowVectorField(lambda p: np.array([-p[1], p[0], 0]), x_range=[-3, 3, 0.6], y_range=[-3, 3, 0.6])
        self.place_in_area(field, 'B2', 'F6', scale_factor=0.45)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(field))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        balloon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/balloon.svg")
        self.place_at_grid(balloon, 'C4', scale_factor=0.7)
        self.play(FadeIn(balloon))
        self.play(balloon.animate.scale(1.5), run_time=1.5)
        self.play(balloon.animate.scale(1/1.5), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        propeller = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/propeller.svg")
        self.place_at_grid(propeller, 'E5', scale_factor=0.7)
        self.play(FadeIn(propeller))
        self.play(Rotate(propeller, angle=PI, run_time=2))
        self.wait(1)

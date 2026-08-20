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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "The power spectrum follows the 5/3 law.",
            "Energy scales with wavenumber to power -5/3.",
            "The inertial subrange ignores viscous effects.",
            "Kolmogorov constant characterizes this universal spectrum.",
            "Turbulence follows clear, predictable mathematical laws."
        ]
        self.setup_layout("The Mathematical Structure: The 5/3 Law", lecture_lines)
        
        # Animations
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        icon_aircraft = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/aircraft.svg")
        self.place_at_grid(icon_aircraft, "B2", scale_factor=0.5)
        self.play(FadeIn(icon_aircraft))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF4500"))
        axes = Axes(x_range=[0, 4], y_range=[0, 4], axis_config={"include_tip": False})
        graph = axes.plot(lambda x: 4 - x, color=WHITE)
        self.place_in_area(axes, "C3", "F5", scale_factor=0.5)
        self.play(Create(axes), Create(graph))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00BFFF"))
        shade = Rectangle(width=1, height=2, color="#00BFFF", fill_opacity=0.3)
        self.place_in_area(shade, "C4", "E4", scale_factor=1)
        self.play(FadeIn(shade))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#ADFF2F"))
        label_c = Text("C", color="#ADFF2F", font_size=24)
        self.place_at_grid(label_c, "B5")
        self.play(Write(label_c))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF00FF"))
        icon_ocean = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ocean.svg")
        self.place_at_grid(icon_ocean, "E2", scale_factor=0.5)
        self.play(FadeIn(icon_ocean), graph.animate.set_color("#FF00FF"))

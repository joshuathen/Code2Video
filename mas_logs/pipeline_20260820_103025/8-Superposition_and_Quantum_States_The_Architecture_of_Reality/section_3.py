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
            "Measurement forces a collapse into single states.",
            "Born rule defines the probability of outcomes.",
            "Square the amplitudes to find the probability.",
            "Think of this as a shrinking cloud.",
            "The interaction picks one definite reality."
        ]
        self.setup_layout("The Measurement Problem & Collapse", lecture_lines)
        
        # Create objects
        cloud = Dot(radius=0.5, color="#FF00FF").set_opacity(0.6)
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/eye.svg]
        measure_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eye.svg", color=WHITE)
        measure_label = Text("Observer", font_size=18, color=WHITE).scale(0.7)
        measure_group = VGroup(measure_icon, measure_label).arrange(DOWN)
        
        # Initial placement as per critique constraints
        self.place_in_area(cloud, "D2", "E3", scale_factor=0.6)
        self.place_at_grid(measure_group, "C5", scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(cloud), FadeIn(measure_group))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        prob_formula = MathTex(r"P = |\alpha|^2", color=YELLOW)
        self.place_at_grid(prob_formula, "E3", scale_factor=0.8)
        self.play(Write(prob_formula))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        # Animate cloud shrinking
        self.play(
            cloud.animate.scale(0.1).set_color("#00FFFF").set_opacity(1.0),
            FadeOut(prob_formula)
        )
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        self.play(
            measure_group.animate.set_color(GREEN),
            Indicate(cloud)
        )
        self.wait(2)

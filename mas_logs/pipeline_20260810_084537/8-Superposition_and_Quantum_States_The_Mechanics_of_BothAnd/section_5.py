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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Real-World Application", [
            "Superposition allows simultaneous processing of possibilities.",
            "Quantum computers explore all paths at once.",
            "Quantum reality is fundamentally 'both/and' logic."
        ])
        
        # Assets
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        self.place_at_grid(computer_icon, "B2", scale_factor=0.5)
        
        # Visuals
        paths = VGroup(*[Line(UP*0.5, DOWN*0.5, color=BLUE).shift(RIGHT*i*0.4) for i in range(-3, 4)])
        # Revised placement based on feedback
        self.place_in_area(paths, "D3", "F5", scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(computer_icon), FadeIn(paths), run_time=1.5)
        self.play(Indicate(paths), run_time=1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        target_path = paths[3]
        self.play(
            FadeOut(VGroup(*[p for p in paths if p != target_path])),
            target_path.animate.set_color(RED),
            run_time=2
        )

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        both_and = Text("'Both/And'", color=WHITE, font_size=40)
        # Revised placement based on feedback
        self.place_at_grid(both_and, "F2", scale_factor=0.8)
        self.play(Write(both_and), run_time=1.5)
        self.wait(2)

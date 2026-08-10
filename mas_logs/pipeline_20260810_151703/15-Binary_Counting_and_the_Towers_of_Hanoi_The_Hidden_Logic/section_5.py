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
        self.setup_layout("Summary and Real-world Application", [
            "Recursion simplifies complex problems.",
            "Binary logic guides efficient systems.",
            "Computers use this for memory management."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Reiterate core recursive rule: 2^n - 1 moves.
        rule_text = MathTex("2^n - 1", " \\text{ moves}", color=YELLOW)
        self.place_in_area(rule_text, 'D2', 'D5', scale_factor=1.2)
        self.play(Write(rule_text))
        self.play(self.lecture[0].animate.set_color("#1ABC9C"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Binary logic summary box.
        summary_box = RoundedRectangle(corner_radius=0.1, height=1.0, width=3.0, color="#1ABC9C")
        summary_text = Text("Binary Logic & Efficiency", font_size=20, color=WHITE)
        summary_group = VGroup(summary_box, summary_text)
        self.place_in_area(summary_group, 'B4', 'B6', scale_factor=0.9)
        
        self.play(FadeIn(summary_group))
        self.play(self.lecture[1].animate.set_color("#2ECC71"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Memory Management + Asset
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        memory_block = Rectangle(height=1.0, width=1.5, color="#9B59B6", fill_opacity=0.3)
        memory_group = VGroup(memory_block, computer_icon).arrange(RIGHT, buff=0.2)
        self.place_in_area(memory_group, 'E2', 'F5', scale_factor=1.0)
        
        self.play(Create(memory_group))
        self.play(self.lecture[2].animate.set_color("#9B59B6"))
        self.wait(2)

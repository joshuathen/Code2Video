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
        lecture_lines = ["Patterns can be deceiving traps.", "Always seek a rigorous proof.", "Don't trust sequences blindly."]
        self.setup_layout("Conclusion: The Power of Skepticism", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Warning icon for "Pattern Trap"
        warning_icon = VGroup(
            RegularPolygon(n=3, color=RED, fill_opacity=0.8),
            Text("!", font_size=40, color=WHITE)
        )
        self.place_in_area(warning_icon, "B2", "D4", scale_factor=0.8)
        self.play(FadeIn(warning_icon))
        self.lecture[0].set_color(RED)

        # === Animation for Lecture Line 2 ===
        # "Proof" text
        proof_text = Text("Proof", font_size=48, color=GREEN)
        self.place_at_grid(proof_text, "C5", scale_factor=1.0)
        self.play(Write(proof_text))
        self.lecture[1].set_color(GREEN)

        # === Animation for Lecture Line 3 ===
        # "Verify" text
        verify_text = Text("Verify", font_size=48, color=TEAL)
        self.place_at_grid(verify_text, "E5", scale_factor=1.0)
        self.play(Write(verify_text))
        self.lecture[2].set_color(TEAL)
        
        self.wait(2)

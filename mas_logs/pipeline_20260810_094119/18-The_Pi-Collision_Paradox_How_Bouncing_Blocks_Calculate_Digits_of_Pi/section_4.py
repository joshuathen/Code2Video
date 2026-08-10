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
        self.setup_layout("The Emergence of Pi", [
            "Collision boundaries form a circular arc.",
            "Paths measure this circular boundary.",
            "Collision counts link to Pi."
        ])
        
        pi_symbol = MathTex(r"\\pi", color=WHITE, font_size=96)
        boundary_circle = Circle(color="#FFA500", radius=1.0)
        
        # Create a container group for better layout management
        pi_group = VGroup(pi_symbol, boundary_circle)
        self.place_in_area(pi_group, 'B2', 'D4', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(pi_symbol))
        self.lecture[0].set_color("#FFA500")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(Create(boundary_circle))
        self.lecture[1].set_color("#FFA500")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(Indicate(pi_symbol, color="#FFFF00"))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)

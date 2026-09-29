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
        self.setup_layout("Summary & Reflection", ["Uncertainty is a fundamental feature.", "Fourier links signals to quantum physics.", "The trade-off is universal."])
        
        # Central text elements
        concept_text = Text("Time and Frequency are reciprocals", font_size=32, color=WHITE)
        grid_title = Text("Fourier Trade-off", font_size=30, color=BLUE)
        grid_group = VGroup(concept_text).arrange(DOWN, buff=0.5)

        # Apply positioning fixes from VideoCritic
        self.place_in_area(concept_text, 'B1', 'C3', scale_factor=0.9)
        self.place_at_grid(grid_title, 'A3', scale_factor=1.2)
        self.place_in_area(grid_group, 'A4', 'F6', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(grid_title), FadeIn(concept_text))
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(concept_text.animate.set_color(YELLOW))
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.wait(2)
        
        self.play(FadeOut(grid_title), FadeOut(concept_text), FadeOut(self.lecture), FadeOut(self.title))

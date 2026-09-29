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
        self.setup_layout("Synthesis & Summary", [
            "Visual anchoring, sequence, and balanced precision.",
            "Explain concepts simply, like to a child.",
            "Transform chaotic numbers into a clear path."
        ])
        
        # Elements
        criteria = VGroup(
            Text("Visual Anchoring", font_size=24),
            Text("Logical Sequence", font_size=24),
            Text("Balanced Precision", font_size=24)
        ).arrange(DOWN, buff=0.4)
        
        # Issue 33: Reposition criteria to B5
        self.place_at_grid(criteria, 'B5', scale_factor=0.7)
        
        # Issue 32: Separate ring and mastery text
        ring = Circle(radius=0.8, color="#FFFF00")
        self.place_at_grid(ring, 'C4', scale_factor=1.2)
        
        # Issue 34: Move mastery text lower
        mastery = Text("Mastery", font_size=32, color="#FF00FF")
        self.place_at_grid(mastery, 'E4', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(criteria), run_time=1.5)
        self.play(self.lecture[0].animate.set_color("#00FFFF"), run_time=0.5)

        # === Animation for Lecture Line 2 ===
        self.play(Create(ring), run_time=1.5)
        self.play(self.lecture[1].animate.set_color("#00FF00"), run_time=0.5)

        # === Animation for Lecture Line 3 ===
        self.play(Write(mastery), run_time=1.5)
        self.play(self.lecture[2].animate.set_color("#FFD700"), run_time=0.5)
        
        self.wait(2)

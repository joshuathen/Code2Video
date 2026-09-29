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
        lecture_lines = ["Topology solves complex discrete problems.", "Existence proofs provide elegant solutions.", "Topology is a tool for fairness."]
        self.setup_layout("Summary and Conclusion", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        topics = VGroup(
            Text("Borsuk-Ulam", color=WHITE),
            Text("Stolen Necklace", color=WHITE),
            Text("Fixed Points", color=WHITE)
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_in_area(topics, 'A1', 'C2', scale_factor=0.6)
        self.play(Write(topics))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        existence_node = Text("Existence Proofs", color="#FF00FF", font_size=36)
        self.place_at_grid(existence_node, 'B3', scale_factor=0.7)
        self.play(FadeIn(existence_node, scale=0.5))
        self.lecture[1].set_color("#FF00FF")

        # === Animation for Lecture Line 3 ===
        fairness_indicator = VGroup(
            Line(LEFT, RIGHT, color="#00FF00"),
            Text("Fairness", color="#00FF00")
        ).arrange(DOWN)
        self.place_at_grid(fairness_indicator, 'B5', scale_factor=0.7)
        self.play(Create(fairness_indicator))
        self.lecture[2].set_color("#00FF00")
        self.wait(2)

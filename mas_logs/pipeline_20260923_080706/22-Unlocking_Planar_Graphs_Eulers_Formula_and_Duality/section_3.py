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
            "Duality builds a new graph structure.",
            "Place a vertex inside every face.",
            "Connect vertices if faces share edges."
        ]
        self.setup_layout("The Concept of Duality", lecture_lines)
        
        # Define base planar graph
        v1 = Dot(color=WHITE)
        v2 = Dot(color=WHITE)
        v3 = Dot(color=WHITE)
        v4 = Dot(color=WHITE)
        
        # Use a container for the graph
        graph = VGroup(
            v1, v2, v3, v4,
            Line(v1.get_center(), v2.get_center(), color=WHITE),
            Line(v2.get_center(), v3.get_center(), color=WHITE),
            Line(v3.get_center(), v4.get_center(), color=WHITE),
            Line(v4.get_center(), v1.get_center(), color=WHITE),
            Line(v1.get_center(), v3.get_center(), color=WHITE)
        )
        
        # Center positions for vertices relative to a target position
        base_pos = self.grid["B4"]
        v1.move_to(base_pos + np.array([-1, 1, 0]))
        v2.move_to(base_pos + np.array([1, 1, 0]))
        v3.move_to(base_pos + np.array([1, -1, 0]))
        v4.move_to(base_pos + np.array([-1, -1, 0]))

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(graph, "B4", scale_factor=0.6)
        self.play(FadeIn(graph))
        self.lecture[0].set_color(BLUE)
        
        # === Animation for Lecture Line 2 ===
        # Place vertices in faces
        f1_dot = Dot(color=YELLOW).move_to(base_pos + np.array([-0.5, 0, 0]))
        f2_dot = Dot(color=YELLOW).move_to(base_pos + np.array([0.5, 0, 0]))
        
        self.play(FadeIn(f1_dot), FadeIn(f2_dot))
        self.lecture[1].set_color(YELLOW)
        
        # === Animation for Lecture Line 3 ===
        # Connect vertices
        connector = Line(f1_dot.get_center(), f2_dot.get_center(), color=RED, stroke_width=4)
        
        self.play(Create(connector))
        self.lecture[2].set_color(RED)
        
        self.wait(2)

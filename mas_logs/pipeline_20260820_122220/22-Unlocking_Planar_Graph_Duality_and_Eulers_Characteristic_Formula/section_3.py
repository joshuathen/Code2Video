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
        self.setup_layout("Concept of the Dual Graph", [
            "Dual graphs place vertices inside each face.",
            "Edges connect vertices of adjacent faces.",
            "This creates a new linked structure."
        ])

        # Primal Graph (Square with diagonal)
        # Fix for issue 26 & 24: Use B2-D4 to balance weight
        v1 = Dot(color=WHITE)
        v2 = Dot(color=WHITE)
        v3 = Dot(color=WHITE)
        v4 = Dot(color=WHITE)

        # Fix for issue 25: Proper grid positioning for primal vertices
        self.place_at_grid(v1, 'C2', scale_factor=0.8)
        self.place_at_grid(v2, 'C5', scale_factor=0.8)
        self.place_at_grid(v3, 'E5', scale_factor=0.8)
        self.place_at_grid(v4, 'E2', scale_factor=0.8)

        edges = VGroup(
            Line(v1.get_center(), v2.get_center(), color=WHITE),
            Line(v2.get_center(), v3.get_center(), color=WHITE),
            Line(v3.get_center(), v4.get_center(), color=WHITE),
            Line(v4.get_center(), v1.get_center(), color=WHITE),
            Line(v1.get_center(), v3.get_center(), color=WHITE)
        )
        primal = VGroup(v1, v2, v3, v4, edges)
        self.add(primal)

        # === Animation for Lecture Line 1 ===
        # Dual vertices
        d1 = Dot(color="#FFFF00")
        d2 = Dot(color="#FFFF00")
        # Position d1 and d2 in the two faces
        d1.move_to(self.grid['B3'] + np.array([0.5, 0, 0]))
        d2.move_to(self.grid['D3'] + np.array([0.5, 0, 0]))
        
        self.play(FadeIn(d1), FadeIn(d2))
        self.lecture[0].set_color("#FFFF00")

        # === Animation for Lecture Line 2 ===
        dual_edge = Line(d1.get_center(), d2.get_center(), color="#FFFF00")
        self.play(Create(dual_edge))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        self.play(
            primal.animate.set_opacity(0.2),
            d1.animate.set_color(RED),
            d2.animate.set_color(RED),
            dual_edge.animate.set_color(RED)
        )
        self.lecture[2].set_color("#FFFF00")

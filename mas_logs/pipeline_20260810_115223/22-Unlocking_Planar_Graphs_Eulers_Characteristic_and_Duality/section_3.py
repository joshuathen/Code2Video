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
        self.setup_layout("Concept of the Dual Graph", ["Construct dual graphs by placing points.", "Place a point in every face.", "Connect points if faces share edges."])
        
        # Primal Graph
        v1 = Dot(color=WHITE)
        v2 = Dot(color=WHITE)
        v3 = Dot(color=WHITE)
        v4 = Dot(color=WHITE)
        
        # Use grid as requested (B2, B5, E2, E5)
        self.place_at_grid(v1, "B2", scale_factor=0.6)
        self.place_at_grid(v2, "B5", scale_factor=0.6)
        self.place_at_grid(v3, "E2", scale_factor=0.6)
        self.place_at_grid(v4, "E5", scale_factor=0.6)
        
        e1 = Line(v1.get_center(), v2.get_center(), color=WHITE)
        e2 = Line(v2.get_center(), v4.get_center(), color=WHITE)
        e3 = Line(v4.get_center(), v3.get_center(), color=WHITE)
        e4 = Line(v3.get_center(), v1.get_center(), color=WHITE)
        e5 = Line(v1.get_center(), v4.get_center(), color=WHITE)
        
        primal = VGroup(v1, v2, v3, v4, e1, e2, e3, e4, e5)
        self.add(primal)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        
        # Dual Vertices - use C3, D3
        dv1 = Dot(color="#FFD700")
        dv2 = Dot(color="#FFD700")
        self.place_at_grid(dv1, "C3", scale_factor=0.5)
        self.place_at_grid(dv2, "D3", scale_factor=0.5)
        self.play(FadeIn(dv1), FadeIn(dv2))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        de = Line(dv1.get_center(), dv2.get_center(), color="#00FF00")
        self.play(Create(de))
        
        self.wait(1)
        
        # Cleanup
        self.play(FadeOut(primal))
        self.play(FadeOut(self.lecture[0]), FadeOut(self.lecture[1]), FadeOut(self.lecture[2]))
        self.play(dv1.animate.set_color("#FFD700"), dv2.animate.set_color("#FFD700"), de.animate.set_color("#FFD700"))
        
        self.wait(2)

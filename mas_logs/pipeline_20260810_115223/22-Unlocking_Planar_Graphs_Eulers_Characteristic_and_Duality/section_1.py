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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: Defining the Planar Graph", [
            "Planar graphs have no intersecting edges.",
            "Identify Vertices as key points.",
            "Edges connect these distinct Vertices.",
            "Faces are regions bounded by Edges.",
            "Consider the infinite exterior region as a face."
        ])
        
        # Create a simple triangle graph
        v1 = Dot(color=WHITE)
        v2 = Dot(color=WHITE)
        v3 = Dot(color=WHITE)
        e1 = Line(v1.get_center(), v2.get_center(), color=WHITE)
        e2 = Line(v2.get_center(), v3.get_center(), color=WHITE)
        e3 = Line(v3.get_center(), v1.get_center(), color=WHITE)
        triangle = VGroup(v1, v2, v3, e1, e2, e3)
        # Fix 25: Reposition triangle to avoid overlap
        self.place_in_area(triangle, 'C3', 'E5', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(triangle), self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        labels = VGroup(
            Text("V=3", font_size=20, color="#FFD700"),
            Text("E=3", font_size=20, color="#FFD700"),
            Text("F=1", font_size=20, color="#FFD700")
        ).arrange(DOWN)
        # Fix 26: Better label placement
        self.place_at_grid(labels, 'C6', scale_factor=0.7)
        self.play(Write(labels), self.lecture[1].animate.set_color("#FFD700"))

        # === Animation for Lecture Line 3 ===
        self.play(e1.animate.set_color("#00FF00"), e2.animate.set_color("#00FF00"), e3.animate.set_color("#00FF00"), self.lecture[2].animate.set_color("#00FF00"))

        # === Animation for Lecture Line 4 ===
        # Fix 21: Asset integration (using SVGMobject)
        asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        cross_graph = VGroup(
            Line(self.grid['C2'], self.grid['E4'], color="#FF4500"),
            Line(self.grid['C4'], self.grid['E2'], color="#FF4500"),
            asset_icon
        )
        self.place_at_grid(asset_icon, 'C3', scale_factor=0.5)
        self.play(FadeIn(cross_graph), self.lecture[3].animate.set_color("#FF4500"))

        # === Animation for Lecture Line 5 ===
        intersection = Dot(color="#FF0000", radius=0.1)
        intersection.move_to(self.grid['D3'])
        self.play(FadeIn(intersection), self.lecture[4].animate.set_color("#FF0000"))
        self.wait(2)

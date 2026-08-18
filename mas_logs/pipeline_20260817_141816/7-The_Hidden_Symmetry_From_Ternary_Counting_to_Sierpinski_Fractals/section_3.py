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
        self.setup_layout("Visualizing the State Space: The Sierpinski Triangle", [
            "Projecting move-states reveals the Sierpinski Triangle structure.",
            "Small disk structures nest within larger recursive patterns.",
            "The optimal solution path traces the fractal edges."
        ])
        
        # Using SVG asset
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"
        
        def get_triangle(color):
            # Using the SVG as a base icon shape, but as a triangle/node placeholder as per storyboard
            return SVGMobject(asset_path, color=color, fill_opacity=0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        tri1 = get_triangle(WHITE)
        self.place_in_area(tri1, 'A4', 'F6', scale_factor=0.4)
        self.play(FadeIn(tri1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GREY)
        self.lecture[1].set_color("#00BFFF")
        
        # Add 3 smaller triangles
        tri2_v = VGroup(
            get_triangle("#00BFFF"),
            get_triangle("#00BFFF"),
            get_triangle("#00BFFF")
        )
        self.place_at_grid(tri2_v[0], 'B3', scale_factor=0.6)
        self.place_at_grid(tri2_v[1], 'D5', scale_factor=0.6)
        self.place_at_grid(tri2_v[2], 'B5', scale_factor=0.6)
        self.play(FadeIn(tri2_v))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GREY)
        self.lecture[2].set_color("#D3D3D3")
        
        # Fill with depth 3 pattern (simplification) using SVG
        pattern = VGroup()
        for i in range(9):
            t = get_triangle("#D3D3D3")
            pattern.add(t)
        self.place_in_area(pattern, 'D1', 'F3', scale_factor=0.5)
        self.play(ReplacementTransform(VGroup(tri1, tri2_v), pattern))
        self.wait(2)

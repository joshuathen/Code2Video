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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Framing the System as Column Vectors", [
            "We represent Ax = b with column vectors.",
            "Vectors act as axes for our target point.",
            "We find weights to scale these vectors."
        ])

        # Assets
        origin_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/origin.svg")
        origin_asset.scale(0.2)
        origin_asset.move_to(self.grid["F1"])
        self.add(origin_asset)

        # Mobjects
        v1 = Vector([1, 2], color="#00FFFF")
        v2 = Vector([2, -1], color="#00FFFF")
        vector_group = VGroup(v1, v2)
        
        v1_label = MathTex(r"v_1", color="#00FFFF")
        v2_label = MathTex(r"v_2", color="#00FFFF")
        
        target_point = Dot(color=WHITE)
        
        # Positions
        self.place_in_area(vector_group, 'A4', 'C6', scale_factor=0.6)
        self.place_at_grid(v1_label, 'A5', scale_factor=0.7)
        self.place_at_grid(v2_label, 'C6', scale_factor=0.7)
        self.place_at_grid(target_point, 'B5', scale_factor=1.2)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Create(v1), Create(v2), Write(v1_label), Write(v2_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        # Extend from origin
        self.play(origin_asset.animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        # Parallelogram
        p1 = v1.get_end()
        p2 = v2.get_end()
        parallelogram = Polygon(ORIGIN, p1, p1 + p2, p2, color="#FFFF00", fill_opacity=0.3)
        self.play(Create(parallelogram))
        self.play(FadeIn(target_point))
        self.wait(2)

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
        # Setup layout
        title = "Visualizing Matrix-Vector Multiplication"
        lines = [
            "Matrix multiplication scales these new basis vectors.",
            "Follow three units of our transformed i-hat vector.",
            "Then add two units of the new j-hat.",
            "Pixel arrives at the final transformed position.",
            "The formula combines these scaled basis vectors."
        ]
        self.setup_layout(title, lines)

        # Define Colors
        i_color = "#58C4DD"
        j_color = "#FC6255"

        # Coordinate Plane setup
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=5,
            y_length=5,
            background_line_style={"stroke_opacity": 0.4}
        )
        self.place_in_area(plane, 'A1', 'E6')
        
        # Transformed basis vectors data
        i_prime = np.array([1, 0.3, 0])
        j_prime = np.array([-0.4, 0.8, 0])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        i_vec = Arrow(plane.c2p(0, 0, 0), plane.c2p(*i_prime), buff=0, color=i_color)
        j_vec = Arrow(plane.c2p(0, 0, 0), plane.c2p(*j_prime), buff=0, color=j_color)
        
        i_label = Text("i'", color=i_color, font_size=18).next_to(i_vec.get_end(), RIGHT, buff=0.1)
        j_label = Text("j'", color=j_color, font_size=18).next_to(j_vec.get_end(), UP, buff=0.1)
        
        self.play(FadeIn(plane))
        self.play(GrowArrow(i_vec), GrowArrow(j_vec), Write(i_label), Write(j_label))
        self.wait(0.5)

        # Scaling vectors (3i' and 2j')
        i_scaled = 3 * i_prime
        j_scaled = 2 * j_prime
        
        i_vec_scaled = Arrow(plane.c2p(0, 0, 0), plane.c2p(*i_scaled), buff=0, color=i_color)
        j_vec_scaled = Arrow(plane.c2p(0, 0, 0), plane.c2p(*j_scaled), buff=0, color=j_color)
        
        i_scaled_label = Text("3i'", color=i_color, font_size=18).next_to(i_vec_scaled.get_end(), RIGHT, buff=0.1)
        j_scaled_label = Text("2j'", color=j_color, font_size=18).next_to(j_vec_scaled.get_end(), LEFT, buff=0.1)

        self.play(
            ReplacementTransform(i_vec, i_vec_scaled),
            ReplacementTransform(j_vec, j_vec_scaled),
            ReplacementTransform(i_label, i_scaled_label),
            ReplacementTransform(j_label, j_scaled_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Break 3i' into units
        seg_i1 = Arrow(plane.c2p(0,0,0), plane.c2p(*i_prime), buff=0, color=i_color, stroke_width=3)
        seg_i2 = Arrow(plane.c2p(*i_prime), plane.c2p(*(2*i_prime)), buff=0, color=i_color, stroke_width=3)
        seg_i3 = Arrow(plane.c2p(*(2*i_prime)), plane.c2p(*(3*i_prime)), buff=0, color=i_color, stroke_width=3)
        
        self.play(FadeOut(i_scaled_label))
        self.play(ReplacementTransform(i_vec_scaled, VGroup(seg_i1, seg_i2, seg_i3)))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Move 2j' segments to tip of 3i'
        j_chained_start = i_scaled
        j_chained_mid = i_scaled + j_prime
        j_chained_end = i_scaled + j_scaled
        
        seg_j1 = Arrow(plane.c2p(*j_chained_start), plane.c2p(*j_chained_mid), buff=0, color=j_color, stroke_width=3)
        seg_j2 = Arrow(plane.c2p(*j_chained_mid), plane.c2p(*j_chained_end), buff=0, color=j_color, stroke_width=3)
        
        self.play(FadeOut(j_scaled_label))
        self.play(ReplacementTransform(j_vec_scaled, VGroup(seg_j1, seg_j2)))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Load Pixel icon asset
        pixel_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/pixel.svg"
        pixel = SVGMobject(pixel_path)
        pixel.set_height(0.3)
        pixel.move_to(plane.c2p(0, 0, 0))
        
        self.play(FadeIn(pixel))
        
        # Define path points for the "weighted sum" traversal
        path_points = [
            plane.c2p(*i_prime),
            plane.c2p(*(2*i_prime)),
            plane.c2p(*(3*i_prime)),
            plane.c2p(*j_chained_mid),
            plane.c2p(*j_chained_end)
        ]
        
        for point in path_points:
            self.play(pixel.animate.move_to(point), run_time=0.5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Formula construction
        # Result components: 3*i' + 2*j' = 3*[1, 0.3] + 2*[-0.4, 0.8] = [3-0.8, 0.9+1.6] = [2.2, 2.5]
        formula = VGroup(
            Text("3", font_size=24, color=i_color),
            Text("i'", font_size=24, color=i_color),
            Text(" + ", font_size=24, color=WHITE),
            Text("2", font_size=24, color=j_color),
            Text("j'", font_size=24, color=j_color),
            Text(" = ", font_size=24, color=WHITE),
            Text("[2.2, 2.5]", font_size=24, color=YELLOW)
        ).arrange(RIGHT, buff=0.1)
        
        # Position formula at the bottom of the grid area spanning F1 to F6
        self.place_in_area(formula, 'F1', 'F6', scale_factor=1.0)
        
        self.play(Write(formula))
        self.wait(2)

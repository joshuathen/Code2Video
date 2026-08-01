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
        # Initial Setup
        title = "Defining a New Basis (B)"
        lines = [
            "Now, imagine a different set of basis vectors.",
            "These new vectors, b1 and b2, create a tilted grid.",
            "The space is the same, but our coordinates change."
        ]
        self.setup_layout(title, lines)

        # --- Coordinate System Preparation ---
        # Standard Grid
        standard_grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=5,
            y_length=5,
            background_line_style={"stroke_color": WHITE, "stroke_opacity": 0.2},
            axis_config={"stroke_color": WHITE, "stroke_opacity": 0.5}
        )
        self.place_in_area(standard_grid, 'A1', 'F6')

        # Basis Vectors b1=(2,1) and b2=(-1,1)
        origin = standard_grid.get_origin()
        b1_end = standard_grid.coords_to_point(2, 1)
        b2_end = standard_grid.coords_to_point(-1, 1)
        
        b1_vec = Arrow(origin, b1_end, buff=0, color="#0000FF")
        b2_vec = Arrow(origin, b2_end, buff=0, color="#FF00FF")
        
        b1_label = Text("b1", weight=BOLD, color="#0000FF")
        b2_label = Text("b2", weight=BOLD, color="#FF00FF")
        
        # Skewed Grid with Clipping logic to prevent obstruction of lecture notes (Issue 35)
        # b1 = [2, 1], b2 = [-1, 1]. Coordinate box is [-4, 4] x [-4, 4].
        skewed_grid = VGroup()
        for i in range(-5, 6):
            # Lines parallel to b2: p(j) = i*b1 + j*b2 = [2i-j, i+j]
            # Solving for j boundaries within [-4, 4] for both components:
            j_min = max(2*i - 4, -4 - i)
            j_max = min(2*i + 4, 4 - i)
            if j_min < j_max:
                p1 = standard_grid.coords_to_point(2*i - j_min, i + j_min)
                p2 = standard_grid.coords_to_point(2*i - j_max, i + j_max)
                skewed_grid.add(Line(p1, p2, color="#555555", stroke_width=1.5, stroke_opacity=0.8))
            
            # Lines parallel to b1: p(k) = k*b1 + i*b2 = [2k-i, k+i]
            k_min = max((i - 4) / 2, -4 - i)
            k_max = min((i + 4) / 2, 4 - i)
            if k_min < k_max:
                p3 = standard_grid.coords_to_point(2*k_min - i, k_min + i)
                p4 = standard_grid.coords_to_point(2*k_max - i, k_max + i)
                skewed_grid.add(Line(p3, p4, color="#555555", stroke_width=1.5, stroke_opacity=0.8))

        # === Animation for Lecture Line 1 ===
        # Line: "Now, imagine a different set of basis vectors."
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(standard_grid), run_time=1)
        self.play(GrowArrow(b1_vec), GrowArrow(b2_vec))
        
        # Grid-based label positioning (Issues 36 and 37)
        self.place_at_grid(b1_label, 'C6', scale_factor=0.8) 
        self.place_at_grid(b2_label, 'B2', scale_factor=0.8) 
        self.play(Write(b1_label), Write(b2_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line: "These new vectors, b1 and b2, create a tilted grid."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(PURPLE)
        )
        self.play(Create(skewed_grid), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: "The space is the same, but our coordinates change."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(GREY)
        )
        self.play(
            standard_grid.animate.set_stroke(opacity=0.05),
            skewed_grid.animate.set_stroke(color=WHITE, opacity=0.4)
        )
        self.wait(2)

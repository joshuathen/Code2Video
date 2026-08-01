from manim import *
import numpy as np

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
        title_text = "The Golden Rule: Row x Column Logic"
        lecture_lines = [
            "In an M-by-N matrix, N is the input dimension.",
            "The number of rows, M, is the output dimension.",
            "Columns dictate how many starting directions we have.",
            "A one-by-three matrix sucks three-D space into one-D.",
            "An entire room collapses into a single infinite line."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors based on storyboard/instruction context
        COLOR_N = "#61AFEF"     # Blue-ish for input
        COLOR_M = "#98C379"     # Green-ish for output
        COLOR_COLS = "#E5C07B"  # Yellow-ish for columns
        COLOR_CLOUD = "#FFFFFF" # White for 3D points
        COLOR_LINE = "#FF00FF"  # Magenta for 1D result

        # === Animation for Lecture Line 1 ===
        # "In an M-by-N matrix, N is the input dimension."
        self.play(self.lecture[0].animate.set_color(COLOR_N))
        
        m_x_n = MathTex("M", "\\times", "N", font_size=48)
        m_x_n[0].set_color(COLOR_M)
        m_x_n[2].set_color(COLOR_N)
        # Fix 31: Use place_in_area for m_x_n
        self.place_in_area(m_x_n, 'B2', 'B4', scale_factor=1.2)
        
        input_label = Text("Input Space (N)", font_size=20, color=COLOR_N)
        self.place_at_grid(input_label, "C4")
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/room.svg
        # Fix 19 & 32: Use room asset and place_in_area
        room_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/room.svg", color=COLOR_N)
        self.place_in_area(room_asset, 'D4', 'E5', scale_factor=1.2)
        
        # Pointing arrow for N
        input_arrow = Arrow(
            m_x_n[2].get_critical_point(DOWN), 
            input_label.get_critical_point(UP), 
            buff=0.1, color=COLOR_N
        )
        
        self.play(
            Write(m_x_n),
            FadeIn(input_label),
            GrowArrow(input_arrow),
            FadeIn(room_asset)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The number of rows, M, is the output dimension."
        self.play(self.lecture[1].animate.set_color(COLOR_M))
        
        output_label = Text("Output Space (M)", font_size=20, color=COLOR_M)
        self.place_at_grid(output_label, "C2")
        
        # Fix 32: Output area M (2D area)
        output_area = Rectangle(width=1.5, height=1.0, color=COLOR_M, fill_opacity=0.3)
        self.place_in_area(output_area, 'D1', 'E2')
        
        output_arrow = Arrow(
            m_x_n[0].get_critical_point(DOWN), 
            output_label.get_critical_point(UP), 
            buff=0.1, color=COLOR_M
        )
        
        self.play(
            FadeIn(output_label),
            GrowArrow(output_arrow),
            Create(output_area)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Columns dictate how many starting directions we have."
        self.play(self.lecture[2].animate.set_color(COLOR_COLS))
        
        # Transition to column logic
        self.play(
            FadeOut(input_label, input_arrow, output_label, output_arrow, room_asset, output_area, m_x_n)
        )
        
        # Build a 1x3 matrix for demonstration
        mat_left = MathTex("[", font_size=40, color=COLOR_COLS)
        mat_right = MathTex("]", font_size=40, color=COLOR_COLS)
        mat_a = MathTex("a", font_size=40, color=COLOR_COLS)
        mat_b = MathTex("b", font_size=40, color=COLOR_COLS)
        mat_c = MathTex("c", font_size=40, color=COLOR_COLS)
        
        matrix_vgroup = VGroup(mat_left, mat_a, mat_b, mat_c, mat_right).arrange(RIGHT, buff=0.3)
        # Fix 31: matrix_vgroup in area B2-B4
        self.place_in_area(matrix_vgroup, 'B2', 'B4', scale_factor=1.2)
        
        # Representative directions (Columns) - X, Y, Z directions
        dir_x = Arrow(ORIGIN, RIGHT, color=RED, buff=0).scale(0.6)
        dir_y = Arrow(ORIGIN, UP, color=GREEN, buff=0).scale(0.6)
        dir_z = Arrow(ORIGIN, [0.7, 0.4, 0], color=BLUE, buff=0).scale(0.6)
        
        dirs = VGroup(dir_x, dir_y, dir_z).arrange(RIGHT, buff=0.8)
        # Fix 33: dirs in area D3-F5
        self.place_in_area(dirs, 'D3', 'F5', scale_factor=1.5)
        
        self.play(Write(matrix_vgroup))
        
        # Link columns to directions using Indicate (L004)
        for i, entry in enumerate([mat_a, mat_b, mat_c]):
            self.play(
                Indicate(entry),
                GrowArrow(dirs[i])
            )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "A one-by-three matrix sucks three-D space into one-D."
        self.play(self.lecture[3].animate.set_color(COLOR_CLOUD))
        
        self.play(FadeOut(matrix_vgroup, dirs))
        
        # Create a 3D-ish cloud
        np.random.seed(123)
        cloud_center = self.grid["D3"]
        dots = VGroup()
        for _ in range(40):
            x = np.random.uniform(-1.5, 1.5)
            y = np.random.uniform(-1.5, 1.5)
            z = np.random.uniform(-1.5, 1.5)
            # Simulated 3D -> 2D projection
            px = x + 0.4 * z
            py = y + 0.3 * z
            dots.add(Dot(point=cloud_center + np.array([px, py, 0]), radius=0.05, color=COLOR_CLOUD))
        
        self.play(FadeIn(dots))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "An entire room collapses into a single infinite line."
        self.play(self.lecture[4].animate.set_color(COLOR_LINE))
        
        # Target 1D infinite line
        line_1d = Line(
            cloud_center + LEFT * 2.5, 
            cloud_center + RIGHT * 2.5, 
            color=COLOR_LINE, 
            stroke_width=4
        )
        
        # Create target line then collapse points
        self.play(Create(line_1d))
        
        collapse_anims = []
        for dot in dots:
            # All dots collapse onto the horizontal line through cloud_center
            target_pos = np.array([dot.get_center()[0], cloud_center[1], 0])
            collapse_anims.append(
                dot.animate.move_to(target_pos).set_color(COLOR_LINE).scale(0.6)
            )
            
        self.play(*collapse_anims, run_time=2, rate_func=rush_into)
        self.wait(2)

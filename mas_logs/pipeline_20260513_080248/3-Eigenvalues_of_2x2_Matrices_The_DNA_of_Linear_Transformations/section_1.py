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
        # Setup layout
        lines = [
            "Meet Leo, our feline friend on a 2D grid.",
            "Matrices transform space, stretching and squashing everything inside.",
            "Watch Leo distort as we apply a linear transformation.",
            "Most points shift to entirely new directions.",
            "But some special lines stay fixed in their orientation."
        ]
        self.setup_layout("Prerequisite & The Core Intuition", lines)

        # Matrix for transformation: [[2, 0.5], [0, 1]]
        matrix = [[2, 0.5], [0, 1]]

        # Asset loading - [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/cat.png]
        leo = ImageMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/cat.png")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Grid area setup
        grid_visual = NumberPlane(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            background_line_style={"stroke_color": "#FFFFFF", "stroke_width": 1, "stroke_opacity": 0.5}
        ).scale(0.5)
        
        self.place_in_area(grid_visual, 'A1', 'F6')
        # Center Leo on the grid (origin)
        self.place_in_area(leo, 'A1', 'F6', scale_factor=0.5)
        
        self.play(Create(grid_visual), FadeIn(leo))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Initial vectors based on the grid's coordinate system
        v_green = Arrow(grid_visual.c2p(0,0), grid_visual.c2p(1,0), color="#00FF00", buff=0, stroke_width=4)
        v_red = Arrow(grid_visual.c2p(0,0), grid_visual.c2p(1,1), color="#FF0000", buff=0, stroke_width=4)
        
        matrix_label = Text("A = [[2, 0.5], [0, 1]]", font_size=20, color=WHITE)
        # Issue 29 Fix: Position matrix label at A4-A5 to avoid clipping
        self.place_in_area(matrix_label, 'A4', 'A5', scale_factor=0.8)

        self.play(GrowArrow(v_green), GrowArrow(v_red), Write(matrix_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Transform the grid, leo, and vectors simultaneously
        # apply_matrix(matrix) shears the space according to linear algebra principles
        self.play(
            grid_visual.animate.apply_matrix(matrix),
            leo.animate.apply_matrix(matrix),
            v_green.animate.apply_matrix(matrix),
            v_red.animate.apply_matrix(matrix),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(RED)
        
        # Highlight red vector changing direction (demonstrates rotation/shear)
        self.play(Indicate(v_red, color="#FF0000"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(GREEN)
        
        # Highlight green vector staying on line (demonstrates eigen-direction)
        eigen_label = Text("Eigen-direction", font_size=18, color="#00FF00")
        # Issue 28 Fix: Position eigen label at E4-E5 to avoid clipping
        self.place_in_area(eigen_label, 'E4', 'E5', scale_factor=0.8)
        
        # Draw a dotted line along the span of the green vector (the eigen-line)
        # We calculate the endpoints based on the transformed vector's position
        vec_dir = v_green.get_end() - v_green.get_start()
        start_pt = v_green.get_start() - vec_dir * 3
        end_pt = v_green.get_start() + vec_dir * 3
        span_line = DashedLine(start_pt, end_pt, color="#00FF00", stroke_opacity=0.5)
        
        self.play(Create(span_line), Write(eigen_label))
        self.play(Indicate(v_green, color="#00FF00"))
        self.wait(2)

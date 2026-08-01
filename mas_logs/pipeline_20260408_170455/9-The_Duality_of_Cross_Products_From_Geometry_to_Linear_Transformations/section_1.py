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
        lecture_lines = [
            'Start with two vectors in two-dimensional space.',
            'Their determinant represents the area of their parallelogram.',
            'Cross products generalize this area intuition into 3D.'
        ]
        self.setup_layout("Prerequisite: The 2D Area Intuition", lecture_lines)
        
        # Colors
        color_u = "#FF5733"
        color_v = "#33FF57"
        color_para = "#FFFF33"
        color_text = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Visual setup
        origin_pos = self.grid["D2"]
        x_axis = Arrow(origin_pos, origin_pos + RIGHT * 3.5, buff=0, color=GRAY_B, stroke_width=2)
        y_axis = Arrow(origin_pos, origin_pos + UP * 3.5, buff=0, color=GRAY_B, stroke_width=2)
        
        # Coordinate definitions
        u_coords = np.array([2.2, 0.6, 0])
        v_coords = np.array([0.8, 1.8, 0])
        
        u_vec = Arrow(origin_pos, origin_pos + u_coords, buff=0, color=color_u)
        v_vec = Arrow(origin_pos, origin_pos + v_coords, buff=0, color=color_v)
        
        # Using Text instead of MathTex to avoid LaTeX FileNotFoundError
        u_label = Text("u", color=color_u, font_size=32, slant=ITALIC)
        v_label = Text("v", color=color_v, font_size=32, slant=ITALIC)
        
        # Fix Issue 27: Move u_label to D6 to avoid overlap
        self.place_at_grid(u_label, "D6", scale_factor=0.8)
        # Fix Issue 26: Move v_label to B4 to avoid overlap
        self.place_at_grid(v_label, "B4", scale_factor=0.8)
        
        self.play(Create(x_axis), Create(y_axis))
        self.play(GrowArrow(u_vec), GrowArrow(v_vec))
        self.play(Write(u_label), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        u_ghost = Arrow(origin_pos + v_coords, origin_pos + v_coords + u_coords, 
                        buff=0, color=color_u, stroke_width=2).set_opacity(0.5)
        v_ghost = Arrow(origin_pos + u_coords, origin_pos + u_coords + v_coords, 
                        buff=0, color=color_v, stroke_width=2).set_opacity(0.5)
        
        para_points = [
            origin_pos,
            origin_pos + u_coords,
            origin_pos + u_coords + v_coords,
            origin_pos + v_coords
        ]
        parallelogram = Polygon(*para_points, color=color_para, fill_color=color_para, fill_opacity=0.3, stroke_width=2)
        
        # Using Text instead of MathTex for the formula
        area_formula = Text("Area = |det(u, v)|", color=color_text, font_size=30)
        # Fix Issue 28: Reposition formula for better space utilization
        self.place_in_area(area_formula, "A3", "B6", scale_factor=0.8)
        
        self.play(
            TransformFromCopy(u_vec, u_ghost),
            TransformFromCopy(v_vec, v_ghost),
            run_time=1.5
        )
        self.play(Create(parallelogram))
        self.play(Write(area_formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        flash_para = parallelogram.copy().set_fill(color_para, opacity=0.7)
        self.play(FadeIn(flash_para), rate_func=there_and_back, run_time=1.5)
        self.wait(2)

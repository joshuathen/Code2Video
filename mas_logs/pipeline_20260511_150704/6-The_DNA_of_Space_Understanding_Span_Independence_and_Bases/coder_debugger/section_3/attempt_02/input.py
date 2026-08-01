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
        # 1. Setup layout
        lecture_lines = [
            'A set is dependent if a vector is redundant.',
            'One vector can be built from the others.',
            'This helper adds no new territory to our span.'
        ]
        self.setup_layout("Linear Dependence: The Redundant Helper", lecture_lines)

        # Shared assets for vector positions
        # Origin at D2
        origin = self.grid['D2']
        x_target = self.grid['B3']
        y_target = self.grid['D4']
        
        vec_x_dir = x_target - origin
        vec_y_dir = y_target - origin
        
        # Define plane (parallelogram) corner
        plane_corner = origin + vec_x_dir + vec_y_dir

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color("#DA70D6"))
        
        # Vectors x and y creating a plane
        vec_x = Arrow(origin, x_target, color="#DA70D6", buff=0)
        vec_y = Arrow(origin, y_target, color="#20B2AA", buff=0)
        
        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        label_x = Text("x", color="#DA70D6", slant=ITALIC)
        self.place_at_grid(label_x, 'B2', scale_factor=0.8)
        
        label_y = Text("y", color="#20B2AA", slant=ITALIC)
        self.place_at_grid(label_y, 'E4', scale_factor=0.8)
        
        # Creating a plane represented by a parallelogram
        plane = Polygon(origin, x_target, plane_corner, y_target, 
                       fill_opacity=0.3, fill_color=WHITE, stroke_width=1, color=WHITE)
        
        self.play(GrowArrow(vec_x), Write(label_x))
        self.play(GrowArrow(vec_y), Write(label_y))
        self.play(FadeIn(plane))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2
        self.play(self.lecture[1].animate.set_color("#20B2AA"))
        
        # Vector z resting exactly on that plane (z = 0.6x + 0.4y)
        z_target = origin + 0.6 * vec_x_dir + 0.4 * vec_y_dir
        vec_z = Arrow(origin, z_target, color="#FF0000", buff=0)
        
        label_z = Text("z", color="#FF0000", slant=ITALIC)
        self.place_at_grid(label_z, 'C3', scale_factor=0.8)
        
        self.play(GrowArrow(vec_z), Write(label_z))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        
        # Flash z and show it is a sum of scaled x and y
        self.play(Indicate(vec_z, color="#FF0000"))
        
        # Show components
        scaled_x_pt = origin + 0.6 * vec_x_dir
        vec_x_comp = Arrow(origin, scaled_x_pt, color="#DA70D6", stroke_width=3, buff=0)
        vec_y_comp = Arrow(scaled_x_pt, z_target, color="#20B2AA", stroke_width=3, buff=0)
        
        self.play(Create(vec_x_comp))
        self.play(Create(vec_y_comp))
        
        self.wait(2)
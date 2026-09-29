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
        lecture_lines = [
            "Visualize a 2D plane under a transformation.",
            "Most vectors change direction as the plane moves.",
            "Some special vectors stay on their original span."
        ]
        self.setup_layout("Visualizing the Concept", lecture_lines)
        
        # Grid and plane
        plane = NumberPlane(x_range=[-3, 3], y_range=[-3, 3], background_line_style={"stroke_opacity": 0.3})
        # Applying requested fix to plane placement
        self.place_in_area(plane, 'A2', 'F4', scale_factor=0.5)
        
        plane_label = Text("Plane", font_size=24, color=WHITE)
        self.place_at_grid(plane_label, 'A2')
        self.add(plane)
        
        # Vector
        vec = Vector(direction=[1, 1, 0], color=PURPLE)
        self.place_at_grid(vec, 'C3', scale_factor=1.0)
        vec_label = Text("v", font_size=24, color=PURPLE)
        self.add(vec_label)
        vec_label.next_to(vec.get_end(), UP)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(plane), Write(plane_label))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        self.play(GrowArrow(vec), FadeIn(vec_label))
        self.play(Rotate(vec, angle=PI/3, about_point=plane.get_center()))
        self.lecture[1].set_color("#FF00FF")
        
        # === Animation for Lecture Line 3 ===
        special_vec = Vector(direction=[2, 0, 0], color=GREEN)
        # Applying requested fixes to special_vec and special_label
        self.place_at_grid(special_vec, 'D2', scale_factor=0.8)
        special_label = Text("Lucky", font_size=24, color=GREEN)
        self.place_at_grid(special_label, 'D3', scale_factor=0.7)
        
        self.play(Create(special_vec), Write(special_label))
        self.play(Transform(special_vec, special_vec.copy().scale(1.5), run_time=2))
        self.lecture[2].set_color("#00FF00")
        
        self.wait(2)

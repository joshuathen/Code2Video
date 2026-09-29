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

class TeachingScene(ThreeDScene):
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

        # Define fine-grained animation grid (6x6 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Shift coordinate system to right side (e.g., x starts from 2.5)
                x = 2.5 + j * 0.7
                y = 2.2 - i * 0.7
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
            "Cross product produces a new vector.",
            "Result is perpendicular to input vectors.",
            "Magnitude equals the parallelogram area.",
            "It defines rotation axes.",
            "Visualization aids conceptual understanding."
        ]
        self.setup_layout("The 3D Cross Product", lecture_lines)
        
        # --- Animation Objects ---
        # 3D object setup
        scene_group = VGroup()
        v1 = Arrow3D(start=ORIGIN, end=RIGHT*1.5, color=BLUE)
        v2 = Arrow3D(start=ORIGIN, end=UP*1.5, color=RED)
        v1_label = Text("u", font_size=20, color=BLUE).next_to(v1.get_end(), RIGHT)
        v2_label = Text("v", font_size=20, color=RED).next_to(v2.get_end(), UP)
        gyro = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gyroscope.svg", color=WHITE)
        
        res_v = Arrow3D(start=ORIGIN, end=OUT*1.5, color="#FF00FF")
        res_label = Text("u x v", font_size=20, color="#FF00FF")
        
        parallelogram = Polygon(ORIGIN, RIGHT*1.5, RIGHT*1.5+UP*1.5, UP*1.5, color="#FFA500", fill_opacity=0.3)
        
        circle = Circle(radius=0.7, color="#00FF00").rotate(PI/2, axis=LEFT).shift(OUT*0.7)
        hinge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hinge.svg", color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_at_grid(gyro, 'A5', scale_factor=0.3)
        self.place_at_grid(v1, 'D4', scale_factor=0.6)
        self.place_at_grid(v2, 'D4', scale_factor=0.6)
        self.add(gyro, v1, v2, v1_label, v2_label)
        self.play(Create(v1), Create(v2), Write(v1_label), Write(v2_label), FadeIn(gyro))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        self.place_at_grid(res_v, 'D4', scale_factor=0.6)
        self.place_at_grid(res_label, 'C5', scale_factor=0.7)
        self.play(Create(res_v), Write(res_label))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFA500")
        para_group = VGroup(parallelogram)
        self.place_in_area(para_group, 'B3', 'E5', scale_factor=0.6)
        self.play(Create(para_group))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        self.place_at_grid(circle, 'D4', scale_factor=0.6)
        self.place_at_grid(hinge, 'B5', scale_factor=0.3)
        self.play(Create(circle), FadeIn(hinge))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        self.set_camera_orientation(phi=60*DEGREES, theta=-45*DEGREES)
        self.wait(2)

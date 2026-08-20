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
        self.setup_layout("Prerequisites: The Anatomy of a Cone and Plane", 
                          ["Visualize a double napped cone meeting at a vertex.", 
                           "A slicing plane cuts through the cone surface.", 
                           "The plane's angle determines the intersection shape."])
        
        # Assets
        cone_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg", color="#FFFFFF")
        plane_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg", color="#00FFFF")

        # === Animation for Lecture Line 1 ===
        # Using SVGMobject per instruction 16
        self.place_at_grid(cone_img, 'B4', scale_factor=0.8)
        self.play(Create(cone_img))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(plane_img, 'B4', scale_factor=1.2)
        self.play(Create(plane_img))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        apex_dot = Dot(color="#FFD700")
        self.place_at_grid(apex_dot, 'B4', scale_factor=0.5)
        apex_label = Text("Vertex", font_size=18, color="#FFD700")
        self.place_at_grid(apex_label, 'D4', scale_factor=0.7)
        self.play(FadeIn(apex_dot), Write(apex_label))
        self.lecture[2].set_color("#FF4500")
        self.wait(2)

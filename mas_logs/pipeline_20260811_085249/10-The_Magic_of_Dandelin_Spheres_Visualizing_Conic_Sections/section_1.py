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
        self.setup_layout("Prerequisites: The Geometry of a Cone", ["A right circular cone extends in two directions.", "A cutting plane slices through the cone.", "This intersection creates a conic section."])
        
        # === Animation for Lecture Line 1 ===
        # Using SVG asset for cone
        cone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg").set_color("#FFFFFF")
        self.place_at_grid(cone, 'C2', scale_factor=0.9)
        self.play(Create(cone))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        plane = Line(LEFT*1.5, RIGHT*1.5).set_color("#FFFF00")
        self.place_at_grid(plane, 'C3', scale_factor=0.8)
        self.play(Create(plane))
        
        label_base = Text("base", font_size=18, color="#FFFF00")
        self.place_at_grid(label_base, 'E4', scale_factor=0.9)
        self.add(label_base)
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        h_line = Line(ORIGIN, DOWN).set_color("#00FFFF")
        r_line = Line(ORIGIN, RIGHT*0.5).set_color("#00FFFF")
        h_label = Text("h", font_size=18, color="#00FFFF")
        r_label = Text("r", font_size=18, color="#00FFFF")
        
        self.place_at_grid(h_line, 'C3')
        self.place_at_grid(r_line, 'C3')
        self.place_at_grid(h_label, 'D2')
        self.place_at_grid(r_label, 'C5')
        
        self.play(Create(h_line), Create(r_line), Write(h_label), Write(r_label))
        self.lecture[2].set_color("#00FFFF")
        self.wait(1)

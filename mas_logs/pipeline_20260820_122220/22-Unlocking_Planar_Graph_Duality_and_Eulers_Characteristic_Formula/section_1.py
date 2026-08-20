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
            "Planar graphs have no crossing edges.",
            "Vertices, edges, and faces define the structure.",
            "Faces include the unbounded outer region."
        ]
        self.setup_layout("Prerequisite: Defining the Planar Graph", lecture_lines)

        # Graph setup - Shifted to right-side grid
        square = Square(side_length=1.5, color=YELLOW)
        self.place_in_area(square, "C4", "E5")
        
        vertices = VGroup(*[Dot(point=p, color=WHITE) for p in square.get_vertices()])
        edges = VGroup(*[Line(square.get_vertices()[i], square.get_vertices()[(i+1)%4], color=YELLOW) for i in range(4)])
        graph = VGroup(vertices, edges)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(Create(graph))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        
        v_label = Text("V", color=WHITE).scale(0.7)
        e_label = Text("E", color=WHITE).scale(0.7)
        
        # Using mandated placement
        self.place_at_grid(v_label, "B4")
        self.place_at_grid(e_label, "D5")
        
        self.play(Write(v_label), Write(e_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE)
        
        face = Polygon(*square.get_vertices(), fill_opacity=0.3, color=RED)
        f_label = Text("F", color=RED).scale(0.7)
        
        # Shifted area
        self.place_in_area(face, "C4", "E5")
        f_label.move_to(face.get_center())
        
        self.play(FadeIn(face), Write(f_label))
        self.wait(2)

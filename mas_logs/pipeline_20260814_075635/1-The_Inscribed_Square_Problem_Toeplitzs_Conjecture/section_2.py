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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Mapping the Geometry to Topology", [
            "Define a square by center and side length.",
            "Map point relations to square symmetry.",
            "Rotate the square to find zero error."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg]
        square = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg")
        square.set_color("#00FFFF")
        center_point = Dot(color=YELLOW)
        square_group = VGroup(square, center_point)
        
        # Layout adjusted per issue 30
        self.place_at_grid(square_group, 'B4', scale_factor=0.6)
        self.play(FadeIn(square_group))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        # Represent point relations to square symmetry
        points = VGroup(*[Dot(color=WHITE) for _ in range(4)])
        # Manually align dots to vertices of square roughly
        v = [square.get_corner(i) for i in [UL, UR, DR, DL]]
        for i, p in enumerate(points):
            p.move_to(v[i])
        
        self.play(Create(points))
        # Rotate square slightly
        self.play(Rotate(square_group, angle=PI/6))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        # Graph of placement error
        axes = Axes(x_range=[0, 3, 1], y_range=[-1, 1, 0.5], axis_config={"include_tip": False})
        error_graph = axes.plot(lambda t: 0.5 * np.sin(t * 2 * PI), color="#FF4500")
        
        graph_group = VGroup(axes, error_graph)
        
        # Layout adjusted per issue 30
        self.place_in_area(graph_group, 'D2', 'F5', scale_factor=0.65)
        self.play(Create(graph_group))
        self.lecture[2].set_color("#FF4500")
        self.wait(2)

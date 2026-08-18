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
        self.setup_layout("Constructing the Peano Curve", [
            "Iterative processes create complex structures.", 
            "We start with a simple cross.", 
            "Subdividing creates increasing detail."
        ])
        
        # Load asset
        none_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        # Axes
        axes = Axes(x_range=[0, 1], y_range=[0, 1], x_length=4, y_length=4, axis_config={"include_numbers": False})
        self.place_in_area(axes, 'B2', 'E5', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.add(axes, none_asset)
        self.place_at_grid(none_asset, "A6", scale_factor=0.2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        
        p0 = axes.c2p(0.5, 0.5)
        p1 = axes.c2p(0.25, 0.5)
        p2 = axes.c2p(0.75, 0.5)
        p3 = axes.c2p(0.5, 0.25)
        p4 = axes.c2p(0.5, 0.75)
        
        cross = VGroup(
            Line(p1, p2, color=YELLOW),
            Line(p3, p4, color=YELLOW)
        )
        self.place_in_area(cross, 'C3', 'D4', scale_factor=0.5)
        self.play(Create(cross))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(ORANGE)
        
        points = VGroup(*[Dot(axes.c2p(x/10, y/10), color=RED, radius=0.03) for x in range(11) for y in range(11)])
        self.place_in_area(points, 'C3', 'E5', scale_factor=0.7)
        
        self.play(Transform(cross, points))
        self.add(none_asset) # Finalize with asset
        self.wait(2)

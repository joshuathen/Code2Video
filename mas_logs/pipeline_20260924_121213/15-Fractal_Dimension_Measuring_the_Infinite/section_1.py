from manim import *

class SierpinskiTriangle(VMobject):
    def __init__(self, order=1, **kwargs):
        super().__init__(**kwargs)
        self.order = order
        points = [UP * 1.5, LEFT * 1.5 + DOWN * 1.5, RIGHT * 1.5 + DOWN * 1.5]
        self.add(*self._get_triangles(points, order))

    def _get_triangles(self, points, order):
        if order == 0:
            return [Polygon(*points, stroke_width=2, fill_opacity=0.5)]
        
        p1, p2, p3 = points
        mid12 = (p1 + p2) / 2
        mid23 = (p2 + p3) / 2
        mid31 = (p3 + p1) / 2
        
        return (self._get_triangles([p1, mid12, mid31], order - 1) +
                self._get_triangles([mid12, p2, mid23], order - 1) +
                self._get_triangles([mid31, mid23, p3], order - 1))

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
            "Euclidean geometry defines standard dimensions.",
            "Points, lines, planes, and cubes exist.",
            "But what about rough, self-similar objects?",
            "Traditional dimensions fail here.",
            "Fractal dimension measures complexity instead."
        ]
        self.setup_layout("The Failure of Traditional Dimensions", lecture_lines)
        
        self.play(self.lecture[0].animate.set_color(BLUE))
        shape_title = Text("Traditional Shapes", font_size=24, color=WHITE)
        self.place_at_grid(shape_title, 'A5', scale_factor=0.9)
        self.play(FadeIn(shape_title))

        self.play(self.lecture[1].animate.set_color(GREEN))
        line = Line(start=LEFT, end=RIGHT, color=BLUE).scale(0.5)
        square = Square(side_length=1.0, color=YELLOW)
        sierpinski = SierpinskiTriangle(order=1).set_color(PURPLE).scale(0.5)
        
        # Adjust placements per review
        self.place_at_grid(line, 'B3', scale_factor=0.7)
        self.place_at_grid(square, 'B4', scale_factor=0.7)
        self.place_at_grid(sierpinski, 'B5', scale_factor=0.7)
        
        self.play(Create(line), Create(square), FadeIn(sierpinski))

        self.play(self.lecture[2].animate.set_color(RED))
        
        self.play(self.lecture[3].animate.set_color(ORANGE))
        cross = Cross(color=RED).scale(0.5)
        cross.move_to(square.get_center())
        self.play(Create(cross))

        self.play(self.lecture[4].animate.set_color(YELLOW))
        self.play(FadeOut(shape_title), FadeOut(line), FadeOut(square), FadeOut(sierpinski), FadeOut(cross))
        text_fdim = Text("Fractal Dimension ~ Complexity", font_size=28, color=YELLOW)
        self.place_in_area(text_fdim, 'D3', 'E5', scale_factor=0.8)
        self.play(Write(text_fdim))
        self.wait(1)

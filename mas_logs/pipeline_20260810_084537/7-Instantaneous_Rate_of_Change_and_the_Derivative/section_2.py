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
        self.setup_layout("The Problem of 'Instantaneous'", [
            "How fast at exactly two seconds?",
            "Zoom in until points nearly coincide.",
            "Secant line approaches the tangent line."
        ])
        
        # Define the function and curve
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 10, 2], axis_config={"include_tip": False}).scale(0.5)
        curve = axes.plot(lambda x: x**2, color=BLUE)
        graph = VGroup(axes, curve)
        
        # Apply fix from issue 22
        self.place_in_area(graph, 'C2', 'F5', scale_factor=0.7)
        
        # Assets
        pointer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pointer.svg")
        magnifier = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        
        # Apply fix from issue 21: P is at x=2, y=4
        p_coords = axes.c2p(2, 4)
        point_p = Dot(p_coords, color="#00FFFF")
        self.place_at_grid(pointer, 'C3', scale_factor=0.5)
        pointer.next_to(point_p, UP)
        
        # secant line definition
        h = ValueTracker(1.5)
        
        def get_secant():
            x1, y1 = 2, 4
            x2, y2 = 2 + h.get_value(), (2 + h.get_value())**2
            line = Line(axes.c2p(x1, y1), axes.c2p(x2, y2), color=YELLOW)
            return line
            
        secant = always_redraw(get_secant)
        tangent = Line(axes.c2p(1, 1), axes.c2p(3, 9), color="#00FF00").scale(0.5).move_to(p_coords)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(graph), Create(point_p), FadeIn(pointer))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.place_at_grid(magnifier, 'C3', scale_factor=0.5)
        self.play(FadeIn(magnifier), FadeOut(pointer))
        self.add(secant)
        self.play(h.animate.set_value(0.1), run_time=3)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(FadeOut(secant), FadeOut(magnifier), Create(tangent))
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        self.wait(2)

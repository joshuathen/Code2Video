from manim import *
import numpy as np

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
        self.setup_layout("Defining Instantaneous Rate (The Tangent Line)", [
            "As time shrinks, points merge together.",
            "The secant line becomes a tangent line.",
            "This represents speed at one exact moment.",
            "We call this the instantaneous rate.",
            "It is the slope at that specific point."
        ])
        
        # Visual setup
        axes = Axes(x_range=[0, 4], y_range=[0, 4], axis_config={"include_tip": False})
        curve = FunctionGraph(lambda x: 0.25 * x**2, x_range=[0, 4], color=BLUE)
        graph = VGroup(axes, curve)
        
        # Asset: Ruler
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        
        # Initial positions
        p1 = Dot(curve.point_from_proportion(0.2), color=WHITE)
        p2 = Dot(curve.point_from_proportion(0.8), color=WHITE)
        secant = Line(p1.get_center(), p2.get_center(), color=RED)
        animation_group = VGroup(p1, p2, secant)

        # Place initially
        self.place_in_area(graph, 'B3', 'F6', scale_factor=0.6)
        self.place_in_area(animation_group, 'B3', 'F6', scale_factor=0.6)
        self.place_at_grid(ruler, 'A6', scale_factor=0.3)
        
        self.add(graph, animation_group, ruler)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        self.play(
            UpdateFromAlphaFunc(p2, lambda m, a: m.move_to(curve.point_from_proportion(0.8 - 0.6 * a))),
            UpdateFromAlphaFunc(secant, lambda m, a: m.put_start_and_end_on(p1.get_center(), curve.point_from_proportion(0.8 - 0.6 * a)))
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        tangent = Line(start=np.array([-1, 0, 0]), end=np.array([1, 0, 0]), color=YELLOW)
        tangent.rotate(np.arctan(0.5 * 0.2))
        tangent.move_to(p1.get_center())
        
        self.play(FadeOut(secant), FadeIn(tangent))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        p1.set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF00FF")
        self.wait(1)

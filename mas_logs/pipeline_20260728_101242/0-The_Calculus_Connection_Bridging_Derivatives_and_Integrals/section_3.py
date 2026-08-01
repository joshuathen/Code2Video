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
        title = "The Integral: Piling Up (Area)"
        lines = [
            "Integrals calculate the total accumulation over time.",
            "Graphically, this is the area under a curve.",
            "Constant speed creates a simple rectangular area.",
            "Varying speeds use Riemann sums to estimate area.",
            "The total shaded area is the distance traveled."
        ]
        self.setup_layout(title, lines)

        # Colors
        VELOCITY_COLOR = "#FFFFFF"
        LIGHT_BLUE_AREA = "#ADD8E6"
        STEEL_BLUE_AREA = "#4682B4"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Setup Axes
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 4, 1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE},
            tips=False
        )
        labels = axes.get_axis_labels(x_label="t", y_label="v(t)")
        graph_group = VGroup(axes, labels)
        
        # Issue 40 Fix: Adjust positioning and scale of graph_group
        self.place_in_area(graph_group, 'B1', 'E6', scale_factor=0.7)
        
        self.play(Write(graph_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        def velocity_fn(t):
            if t <= 2:
                return 2
            else:
                return 2 + np.sin(t - 2) * (t - 2) * 0.5

        curve = axes.plot(velocity_fn, x_range=[0, 5], color=VELOCITY_COLOR)
        area = axes.get_area(curve, x_range=[0, 5], color=LIGHT_BLUE_AREA, opacity=0.5)
        
        self.play(Create(curve))
        self.play(FadeIn(area))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        rect_area = axes.get_area(curve, x_range=[0, 2], color=YELLOW, opacity=0.7)
        rect_label = Text("Rectangle", font_size=18, color=YELLOW)
        
        # Issue 42 Fix: Adjust positioning and scale of rect_label
        self.place_at_grid(rect_label, 'B2', scale_factor=0.8)

        self.play(FadeIn(rect_area), Write(rect_label))
        self.wait(1)
        self.play(FadeOut(rect_area), FadeOut(rect_label))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        self.play(FadeOut(area))

        riemann_sums = axes.get_riemann_rectangles(
            curve, x_range=[2, 5], dx=0.5, color=LIGHT_BLUE_AREA, input_sample_type="left"
        )
        
        self.play(Create(riemann_sums))
        self.wait(1)

        finer_riemann = axes.get_riemann_rectangles(
            curve, x_range=[2, 5], dx=0.1, color=LIGHT_BLUE_AREA, input_sample_type="left"
        )
        self.play(Transform(riemann_sums, finer_riemann))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        solid_area = axes.get_area(curve, x_range=[0, 5], color=STEEL_BLUE_AREA, opacity=0.8)
        
        # Issue 32 Fix: Use provided ruler asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg]
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        ruler.set_color(WHITE)
        self.place_in_area(ruler, 'F1', 'F6', scale_factor=0.8)
        
        distance_label = Text("Distance Traveled", font_size=20, color=STEEL_BLUE_AREA)
        
        # Issue 41 Fix: Adjust positioning and scale of distance_label
        self.place_at_grid(distance_label, 'F3', scale_factor=0.8)

        self.play(
            FadeOut(riemann_sums),
            FadeIn(solid_area),
            FadeIn(ruler),
            FadeIn(distance_label)
        )
        self.wait(2)

        self.lecture[4].set_color(WHITE)
        self.wait(1)

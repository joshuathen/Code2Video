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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'The forward process follows a predictable Markov Chain.',
            'We add noise incrementally using a variance schedule, beta.',
            'Each step slowly dissolves structure into mathematical chaos.',
            'This provides the ground truth for our neural network.'
        ]
        self.setup_layout("The Forward Process: Order to Chaos", lecture_lines)
        
        # Colors
        COLOR_TRANSITION = "#00FFFF"
        COLOR_BETA = "#FFA500"
        COLOR_IMAGE = "#33CCFF"
        ASSET_IMAGE = "/mmfs1/data/home/jthen/Code2Video/assets/icon/image.svg"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Create 5 boxes representing x0 to xT
        boxes = VGroup()
        labels = VGroup()
        box_indices = ["0", "1", "t-1", "t", "T"]
        grid_cols = ["1", "2", "3", "4", "6"]
        
        for i, idx in enumerate(box_indices):
            # Box frame
            rect = Rectangle(width=0.8, height=0.8, color=WHITE, stroke_width=2)
            self.place_at_grid(rect, f"B{grid_cols[i]}")
            
            # Content: Image Asset + Gradually add noise dots
            content = VGroup()
            num_dots = [0, 25, 60, 110, 220][i]
            img_opacity = [1.0, 0.7, 0.4, 0.2, 0.0][i]
            
            if img_opacity > 0:
                try:
                    img_icon = SVGMobject(ASSET_IMAGE).set_color(COLOR_IMAGE)
                    img_icon.set_width(0.6)
                    img_icon.set_opacity(img_opacity)
                    img_icon.move_to(rect.get_center())
                    content.add(img_icon)
                except:
                    # Fallback if SVG missing in environment
                    fallback = Square(side_length=0.6, fill_opacity=img_opacity, fill_color=COLOR_IMAGE)
                    fallback.move_to(rect.get_center())
                    content.add(fallback)
            
            if num_dots > 0:
                dots = VGroup(*[
                    Dot(
                        point=rect.get_center() + np.array([np.random.uniform(-0.35, 0.35), np.random.uniform(-0.35, 0.35), 0]),
                        radius=0.012,
                        color=WHITE
                    ) for _ in range(num_dots)
                ])
                content.add(dots)
            
            boxes.add(VGroup(rect, content))
            
            # Label below
            label = Text(f"x{idx}", font_size=18)
            label.next_to(rect, DOWN, buff=0.1)
            labels.add(label)

        # Ellipsis between t and T
        dots_ellipsis = Text("...", font_size=24)
        # ISSUE 44 / 57 Fix: scale_factor=0.6
        self.place_at_grid(dots_ellipsis, "B5", scale_factor=0.6)

        # Animate boxes appearance
        self.play(LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.2), run_time=1.5)
        self.play(Write(labels), Write(dots_ellipsis))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_BETA))
        
        # Variance Schedule Graph
        axes_origin = self.grid["F1"] + LEFT*0.4 + DOWN*0.2
        x_axis = Line(axes_origin, self.grid["F6"] + RIGHT*0.2, color=GRAY)
        y_axis = Line(axes_origin, self.grid["D1"] + UP*0.2, color=GRAY)
        
        graph_labels = VGroup(
            Text("t", font_size=16).next_to(x_axis, RIGHT, buff=0.1),
            Text("βₜ", font_size=16, color=COLOR_BETA).next_to(y_axis, UP, buff=0.1)
        )
        
        # Curve: Increasing variance schedule
        curve_points = []
        for x_val in np.linspace(0, 4.5, 20):
            # Quadratic growth
            y_val = (x_val**2) / 10.0
            curve_points.append(axes_origin + np.array([x_val, y_val, 0]))
        
        beta_curve = VMobject(color=COLOR_BETA)
        beta_curve.set_points_as_corners(curve_points)
        
        self.play(Create(x_axis), Create(y_axis), Write(graph_labels))
        self.play(Create(beta_curve), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_TRANSITION))
        
        # Highlight transition between x_{t-1} and x_t
        box_t_minus_1 = boxes[2][0]
        box_t = boxes[3][0]
        
        arrow = Arrow(
            start=box_t_minus_1.get_right() + RIGHT*0.1,
            end=box_t.get_left() + LEFT*0.1,
            buff=0,
            color=COLOR_TRANSITION,
            stroke_width=4
        )
        
        # Markov formula: q(xt | xt-1)
        formula = Text("q(xₜ | xₜ₋₁)", font_size=20, color=COLOR_TRANSITION)
        formula.next_to(arrow, UP, buff=0.1)
        
        glow = arrow.copy().set_stroke(width=8, opacity=0.3)
        
        self.play(
            GrowArrow(arrow),
            FadeIn(glow),
            Write(formula)
        )
        self.play(Indicate(formula, color=COLOR_TRANSITION, scale_factor=1.1))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(WHITE))
        
        # Final emphasis: connect x0 and xT
        full_path_rect = Rectangle(color=WHITE, stroke_width=1).surround(boxes, buff=0.2)
        truth_label = Text("Training Target", font_size=22, color=WHITE)
        # ISSUE 43 / 57 Fix: Position in area A1 to A2 with scale 0.9
        self.place_in_area(truth_label, "A1", "A2", scale_factor=0.9)
        
        self.play(Create(full_path_rect))
        self.play(Write(truth_label))
        self.play(Circumscribe(boxes[0], color=COLOR_IMAGE))
        
        self.wait(2)

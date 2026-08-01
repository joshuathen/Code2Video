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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initial layout
        lines = [
            'Meet Nero, a robot learning to recognize images.',
            'Inside Nero are adjustable knobs called weights.',
            'Nero sees a dog but incorrectly guesses muffin.',
            'This mismatch creates an error signal to learn.',
            'We must adjust knobs to get correct guesses.'
        ]
        self.setup_layout("The Big Picture: Nero the Robot Learns to See", lines)

        # === Animation for Lecture Line 1 ===
        # Meet Nero, a robot learning to recognize images.
        self.lecture[0].set_color("#ADD8E6")
        
        # Nero Robot Asset
        # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg]
        nero = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg")
        nero.set_color("#ADD8E6")
        self.place_in_area(nero, 'A1', 'B2', scale_factor=0.8)

        # Input Image Grid representing data
        image_grid = VGroup(*[
            Square(side_length=0.2, color=GRAY, fill_opacity=0.1) for _ in range(9)
        ]).arrange_in_grid(rows=3, cols=3, buff=0.05)
        for i, sq in enumerate(image_grid):
            if i % 2 == 0: sq.set_fill(WHITE, opacity=0.8)
        self.place_at_grid(image_grid, 'C1', scale_factor=0.8) # Issue 44: Scale 0.8
        
        self.play(FadeIn(nero), FadeIn(image_grid))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Inside Nero are adjustable knobs called weights.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#C0C0C0")
        
        # Network inside Nero's brain (represented as nodes and knobs)
        node_in = Circle(radius=0.15, color=WHITE)
        self.place_at_grid(node_in, 'C2')
        node_h1 = Circle(radius=0.15, color=WHITE)
        self.place_at_grid(node_h1, 'B3')
        node_h2 = Circle(radius=0.15, color=WHITE)
        self.place_at_grid(node_h2, 'D3')
        node_out = Circle(radius=0.15, color=WHITE)
        self.place_at_grid(node_out, 'C4')
        nodes = VGroup(node_in, node_h1, node_h2, node_out)

        # Edges (Weights) with grey knobs (#C0C0C0)
        edge1 = Line(node_in.get_right(), node_h1.get_left(), color=GRAY)
        edge2 = Line(node_in.get_right(), node_h2.get_left(), color=GRAY)
        edge3 = Line(node_h1.get_right(), node_out.get_left(), color=GRAY)
        edge4 = Line(node_h2.get_right(), node_out.get_left(), color=GRAY)
        edges = VGroup(edge1, edge2, edge3, edge4)

        def get_knob(edge):
            center = edge.get_center()
            knob_circ = Circle(radius=0.1, color="#C0C0C0", fill_opacity=0.3).move_to(center)
            knob_line = Line(knob_circ.get_center(), knob_circ.get_top(), color="#C0C0C0")
            return VGroup(knob_circ, knob_line)

        knobs = VGroup(
            get_knob(edge1), get_knob(edge2), get_knob(edge3), get_knob(edge4)
        )

        self.play(Create(edges), Create(nodes))
        self.play(FadeIn(knobs))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Nero sees a dog but incorrectly guesses muffin.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF0000")
        
        # Data pulses (#FFFFFF) flowing to output
        pulse1 = Dot(radius=0.08, color=WHITE).move_to(node_in.get_center())
        pulse2 = Dot(radius=0.08, color=WHITE).move_to(node_in.get_center())
        
        muffin_label = Text("Muffin", font_size=18, color="#FF0000")
        self.place_at_grid(muffin_label, 'B5')

        self.play(
            pulse1.animate.move_to(node_h1.get_center()),
            pulse2.animate.move_to(node_h2.get_center()),
            run_time=0.8
        )
        self.play(
            pulse1.animate.move_to(node_out.get_center()),
            pulse2.animate.move_to(node_out.get_center()),
            run_time=0.8
        )
        self.play(
            node_out.animate.set_color("#FF0000").set_fill("#FF0000", opacity=0.5),
            Write(muffin_label),
            FadeOut(pulse1), FadeOut(pulse2)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This mismatch creates an error signal to learn.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FF4500")

        # Truth: Dog [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/dog.svg]
        dog_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/dog.svg")
        dog_asset.set_color(WHITE)
        truth_text = Text("Truth: Dog", font_size=16, color=WHITE)
        truth_group = VGroup(dog_asset, truth_text).arrange(RIGHT, buff=0.2)
        self.place_at_grid(truth_group, 'D5', scale_factor=0.6)
        
        # Error indicators: Red X-mark and large red Error bar
        x_mark = VGroup(
            Line(UP+LEFT, DOWN+RIGHT, color=RED),
            Line(UP+RIGHT, DOWN+LEFT, color=RED)
        ).scale(0.2)
        self.place_at_grid(x_mark, 'D6') # Issue 43: D6

        error_bar_bg = Line(self.grid['B5'], self.grid['D5'], color=GRAY, stroke_width=2)
        error_bar = Line(self.grid['B5'], self.grid['D5'], color="#FF4500", stroke_width=6)
        error_text = Text("Error", font_size=16, color="#FF4500")
        self.place_at_grid(error_text, 'C6', scale_factor=0.8) # Issue 42: C6

        self.play(FadeIn(truth_group), Create(x_mark))
        self.play(Create(error_bar_bg), GrowFromCenter(error_bar))
        self.play(Write(error_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # We must adjust knobs to get correct guesses.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFFFF")

        # Flash the grey knobs in bright white to signal adjustment
        self.play(knobs.animate.set_color(WHITE), run_time=0.4)
        self.play(knobs.animate.set_color("#C0C0C0"), run_time=0.4)
        
        # Adjustment animation: Rotate knobs and shrink error bar
        self.play(
            *[Rotate(k[1], angle=PI/2) for k in knobs],
            error_bar.animate.scale(0.1),
            error_text.animate.set_opacity(0.3),
            run_time=2
        )
        
        self.wait(2)

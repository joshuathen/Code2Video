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

class Section2PrerequisitesScene(TeachingScene):
    def construct(self):
        title = "Prerequisite: The Anatomy of a Decision"
        lines = [
            "Networks use weights as adjustable control knobs.",
            "High weights mean an input is very important.",
            "A bias sets the baseline for every decision."
        ]
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        # Networks use weights as adjustable control knobs.
        self.lecture[0].set_color(YELLOW)
        
        # Circles representing 'Floppy Ears' and 'Metal Texture' (#FFFFFF)
        fe_circle = Circle(radius=0.35, color=WHITE)
        mt_circle = Circle(radius=0.35, color=WHITE)
        self.place_at_grid(fe_circle, 'B1')
        self.place_at_grid(mt_circle, 'E1')
        
        fe_label = Text("Floppy Ears", font_size=16, color=WHITE)
        mt_label = Text("Metal Texture", font_size=16, color=WHITE)
        self.place_at_grid(fe_label, 'A1')
        self.place_at_grid(mt_label, 'F1')
        
        # Central neuron (#00FFFF)
        neuron = Circle(radius=0.5, color="#00FFFF", fill_opacity=0.3)
        self.place_at_grid(neuron, 'C4')
        n_label = Text("Neuron", font_size=16, color="#00FFFF")
        self.place_at_grid(n_label, 'B4')
        
        # Flow arrows toward a central neuron
        arrow_fe = Line(fe_circle.get_right(), neuron.get_left(), buff=0.1, color=WHITE).add_tip()
        arrow_mt = Line(mt_circle.get_right(), neuron.get_left(), buff=0.1, color=WHITE).add_tip()
        
        # Knobs (#A9A9A9) appear on the arrows
        knob_fe_base = Circle(radius=0.15, color="#A9A9A9", fill_opacity=0.5)
        knob_fe_line = Line(ORIGIN, UP * 0.15, color="#A9A9A9")
        knob_fe = VGroup(knob_fe_base, knob_fe_line)
        self.place_at_grid(knob_fe, 'B2')
        
        knob_mt_base = Circle(radius=0.15, color="#A9A9A9", fill_opacity=0.5)
        knob_mt_line = Line(ORIGIN, UP * 0.15, color="#A9A9A9")
        knob_mt = VGroup(knob_mt_base, knob_mt_line)
        self.place_at_grid(knob_mt, 'E2')
        
        self.play(
            FadeIn(fe_circle), FadeIn(mt_circle),
            Write(fe_label), Write(mt_label)
        )
        self.play(GrowFromCenter(neuron), Write(n_label))
        self.play(Create(arrow_fe), Create(arrow_mt))
        self.play(FadeIn(knob_fe), FadeIn(knob_mt))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # High weights mean an input is very important.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # The 'Floppy Ears' knob turns to a high setting (#FFFF00)
        self.play(
            knob_fe.animate.set_color("#FFFF00"),
            knob_fe_line.animate.rotate(-PI/2, about_point=knob_fe_base.get_center()),
            arrow_fe.animate.set_stroke(color="#FFFF00", width=8)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A bias sets the baseline for every decision.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # A 'Bias' slider (#FFA500) moves
        slider_track = Line(self.grid['A5'], self.grid['B5'], color="#FFA500")
        slider_knob = Square(side_length=0.15, color="#FFA500", fill_opacity=1)
        slider_knob.move_to(slider_track.get_start())
        bias_label = Text("Bias", font_size=16, color="#FFA500")
        self.place_at_grid(bias_label, 'A4')
        
        # Neuron's output bulb (#FFFFFF) glows brighter
        bulb = Circle(radius=0.25, color=WHITE, fill_opacity=0.1)
        self.place_at_grid(bulb, 'C6')
        bulb_label = Text("Output", font_size=16, color=WHITE)
        self.place_at_grid(bulb_label, 'B6')
        
        self.play(
            Create(slider_track),
            FadeIn(slider_knob),
            Write(bias_label),
            FadeIn(bulb),
            Write(bulb_label)
        )
        
        # Slider moves and bulb glows
        self.play(
            slider_knob.animate.move_to(slider_track.get_end()),
            bulb.animate.set_fill(WHITE, opacity=1).scale(1.5),
            neuron.animate.set_fill(opacity=0.8),
            run_time=2
        )
        self.wait(2)

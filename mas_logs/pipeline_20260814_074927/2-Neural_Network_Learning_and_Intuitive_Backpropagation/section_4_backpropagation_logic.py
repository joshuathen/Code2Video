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

class Section4BackpropagationLogicScene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "To improve, we send error signals backward.",
            "We ask which knobs caused the final mistake.",
            "The chain rule connects every layer's influence.",
            "Each connection gets a share of the blame.",
            "This blame guides the direction of change."
        ]
        self.setup_layout("Backpropagation: Tracing the Blame", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # A red glow (#FF0000) starts at the output lightbulb and pulses back toward the neuron.
        self.lecture[0].set_color("#FF0000")
        
        # Output lightbulb at D6
        lightbulb_circle = Circle(radius=0.3, color=WHITE, fill_opacity=0.2)
        lightbulb_rays = VGroup(*[
            Line(0.3 * np.array([np.cos(a), np.sin(a), 0]), 0.5 * np.array([np.cos(a), np.sin(a), 0]))
            for a in np.linspace(0, 2*PI, 8, endpoint=False)
        ])
        lightbulb = VGroup(lightbulb_circle, lightbulb_rays)
        self.place_at_grid(lightbulb, "D6")
        
        # Neuron at D3
        neuron = Circle(radius=0.3, color=BLUE, fill_opacity=0.1)
        neuron_label = Text("Metal Texture", font_size=16, color=BLUE).next_to(neuron, DOWN, buff=0.1)
        neuron_group = VGroup(neuron, neuron_label)
        self.place_at_grid(neuron_group, "D3")
        
        connection = Line(self.grid["D3"], self.grid["D6"], color=GRAY)
        
        # Red glow pulse
        glow = Dot(color="#FF0000", radius=0.15)
        glow.move_to(self.grid["D6"])
        
        self.add(lightbulb, neuron_group, connection)
        self.play(FadeIn(glow))
        self.play(glow.animate.move_to(self.grid["D3"]), run_time=1.5)
        self.play(Flash(self.grid["D3"], color="#FF0000", flash_radius=0.5), FadeOut(glow))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The connection for 'Metal Texture' flashes red (#FF0000) to indicate high blame.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF0000")
        
        knob = VGroup(
            Circle(radius=0.15, color=WHITE, fill_opacity=0.5),
            Line(ORIGIN, UP*0.15, color=RED)
        )
        # Using D4 as midpoint.
        self.place_at_grid(knob, "D4")
        
        self.play(Create(knob))
        self.play(connection.animate.set_color("#FF0000"), knob.animate.set_color("#FF0000"))
        self.play(Indicate(connection, color="#FF0000"), Indicate(knob, color="#FF0000"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Interlocked gears (#D3D3D3) rotate. One gear's movement affects the next in sequence.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#D3D3D3")
        
        def create_gear(radius=0.35, teeth=8, color="#D3D3D3"):
            gear = VGroup(Circle(radius=radius, color=color))
            for i in range(teeth):
                tooth = Rectangle(width=0.12, height=0.12, color=color, fill_opacity=1)
                angle = i * (2 * PI / teeth)
                tooth.move_to(radius * np.array([np.cos(angle), np.sin(angle), 0]))
                tooth.rotate(angle)
                gear.add(tooth)
            return gear

        gear1 = create_gear()
        gear2 = create_gear()
        gear3 = create_gear()
        
        self.place_at_grid(gear1, "B2")
        self.place_at_grid(gear2, "B3")
        self.place_at_grid(gear3, "B4")
        
        self.play(FadeIn(gear1), FadeIn(gear2), FadeIn(gear3))
        
        # Persistent rotation via updaters
        gear1.add_updater(lambda m, dt: m.rotate(dt * PI/2))
        gear2.add_updater(lambda m, dt: m.rotate(-dt * PI/2))
        gear3.add_updater(lambda m, dt: m.rotate(dt * PI/2))
        
        self.wait(3)
        # Remove updaters after rotation session to keep performance
        gear1.clear_updaters()
        gear2.clear_updaters()
        gear3.clear_updaters()
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Backward arrows (#FF4500) carry 'correction' values to each knob.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FF4500")
        
        arrow1 = Arrow(start=self.grid["D6"], end=self.grid["D4"], color="#FF4500", buff=0.1)
        arrow2 = Arrow(start=self.grid["D4"], end=self.grid["D3"], color="#FF4500", buff=0.1)
        
        self.play(GrowArrow(arrow1))
        self.play(GrowArrow(arrow2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Numbers like -0.5 and +0.2 (#FFFFFF) appear next to the connection lines.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFFFF")
        
        val1 = Text("-0.5", font_size=18, color=WHITE)
        val2 = Text("+0.2", font_size=18, color=WHITE)
        
        # Place near the arrows
        self.place_at_grid(val1, "C4", scale_factor=1.0)
        self.place_at_grid(val2, "C3", scale_factor=1.0)
        
        self.play(Write(val1), Write(val2))
        self.wait(2)

        # Final cleanup
        self.lecture[4].set_color(WHITE)
        self.wait(1)

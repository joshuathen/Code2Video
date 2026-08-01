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

class Section6Scene(TeachingScene):
    def construct(self):
        # 1. Setup layout
        title_text = "The Music of Primes: Conclusion"
        lecture_lines = [
            "Each zero acts like a musical note in an orchestra.",
            "Together, they create a wave describing prime number density.",
            "The Zeta function reveals the hidden rhythm of the primes."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        NOTE_COLOR = "#E0FFFF"   # Light cyan (glowing notes)
        WAVE_COLOR = "#00FFFF"   # Cyan (sum wave)
        STAIRCASE_COLOR = "#FFD700" # Gold (prime distribution)
        ZERO_COLOR = "#FFFFFF"   # White (initial zeros)
        ASSET_PATH = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/musical.svg"

        # === Animation for Lecture Line 1 ===
        # Line 1: "Each zero acts like a musical note in an orchestra."
        self.lecture[0].set_color(NOTE_COLOR)
        
        # Initial zero points on the critical line
        zero_positions = ["B3", "C3", "D3", "E3", "F3"]
        zeros_dots = VGroup(*[Dot(color=ZERO_COLOR, radius=0.08) for _ in zero_positions])
        for i, pos in enumerate(zero_positions):
            self.place_at_grid(zeros_dots[i], pos)
            
        critical_line = Line(self.grid["A3"], self.grid["F3"], color=BLUE_E, stroke_opacity=0.6)
        
        self.play(Create(critical_line), FadeIn(zeros_dots, shift=RIGHT))
        self.wait(0.5)
        
        # Transform zeros into musical notes [Asset: musical.svg]
        musical_notes = VGroup()
        for dot in zeros_dots:
            note = SVGMobject(ASSET_PATH).scale(0.25).set_color(NOTE_COLOR).move_to(dot)
            musical_notes.add(note)
            
        glows = VGroup(*[
            Circle(radius=0.2, color=NOTE_COLOR, fill_opacity=0.2, stroke_width=0).move_to(dot)
            for dot in zeros_dots
        ])
        
        self.play(
            ReplacementTransform(zeros_dots, musical_notes),
            FadeIn(glows),
            critical_line.animate.set_stroke(opacity=0.2)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Line 2: "Together, they create a wave describing prime number density."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(WAVE_COLOR)
        
        # Create a complex interference wave
        def interference_wave(x):
            # Sum of sin waves to represent the fluctuation term in prime distribution
            return 0.5 * (np.sin(3*x) + 0.5*np.sin(7*x) + 0.3*np.sin(13*x))

        wave_graph = FunctionGraph(
            interference_wave, 
            x_range=[0, 4], 
            color=WAVE_COLOR,
            stroke_width=4
        )
        # Move wave to specified area per Issue 49
        self.place_in_area(wave_graph, "B3", "E6", scale_factor=1.0)
        
        # Transition: Musical notes merge into the wave
        self.play(
            ReplacementTransform(VGroup(musical_notes, glows), wave_graph),
            FadeOut(critical_line),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Line 3: "The Zeta function reveals the hidden rhythm of the primes."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(STAIRCASE_COLOR)
        
        # The Prime-Counting Function staircase (pi(x))
        # Stylized staircase to illustrate the concept
        stair_pts = []
        curr_x, curr_y = -2, -1
        for i in range(10):
            step_width = 0.4 + 0.2*np.random.rand()
            stair_pts.append([curr_x, curr_y, 0])
            curr_x += step_width
            stair_pts.append([curr_x, curr_y, 0])
            curr_y += 0.3
            
        staircase = VMobject(color=STAIRCASE_COLOR, stroke_width=5)
        staircase.set_points_as_corners(stair_pts)
        
        # Place staircase in specified area per Issue 49
        self.place_in_area(staircase, "B3", "E6", scale_factor=1.0)
        
        # Align wave graph to overlay correctly on the staircase profile
        # We'll slightly shift it to emphasize the "revealing rhythm" aspect
        self.play(
            FadeIn(staircase, shift=UP),
            wave_graph.animate.shift(UP * 0.2)
        )
        self.wait(1)
        
        # Final highlight
        self.play(
            Indicate(staircase, scale_factor=1.1, color=STAIRCASE_COLOR),
            Indicate(wave_graph, scale_factor=1.1, color=WAVE_COLOR),
            run_time=2
        )
        self.wait(3)

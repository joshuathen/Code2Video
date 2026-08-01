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
        # Setup the scene layout
        self.setup_layout(
            "The Big Idea: The 'Echo' Effect", 
            [
                'Convolution describes how one function modifies another.', 
                'Imagine a signal passing through a system with memory.', 
                'A Cyber-Frog jumps on a quiet digital pond.', 
                'Each splash creates ripples that fade over time.', 
                'Total state equals the sum of all past ripples.'
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        center_title = Text("Discrete Convolution", font_size=42, color=WHITE)
        # Resolved Issue 25: Move title to row B to prevent overlap with ripples
        self.place_in_area(center_title, 'B1', 'B6', scale_factor=0.8)
        self.play(FadeIn(center_title))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(FadeOut(center_title))
        
        # Draw the "pond" (a blue baseline)
        pond_line = Line(self.grid["E1"], self.grid["E6"], color="#0000FF", stroke_width=4)
        self.add(pond_line)
        self.play(Create(pond_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Cyber-frog (Yellow circle)
        frog = Circle(radius=0.15, color="#FFFF00", fill_opacity=1.0)
        # Resolved Issue 26: Place frog in centered area E1-F6
        self.place_in_area(frog, 'E1', 'F6', scale_factor=1.0)
        
        # Jump 1 path
        jump_path1 = ArcBetweenPoints(start=self.grid["E1"], end=self.grid["E2"], angle=-PI/2)
        
        self.play(FadeIn(frog))
        self.play(MoveAlongPath(frog, jump_path1), run_time=1)
        
        # White spike at jump site
        spike1 = Line(self.grid["E2"], self.grid["E2"] + UP*0.8, color=WHITE, stroke_width=6)
        self.play(Create(spike1))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Blue ripples (bars) decreasing in height
        ripple1_set = VGroup(
            Line(self.grid["E2"], self.grid["E2"] + UP*0.6, color="#0000FF", stroke_width=10),
            Line(self.grid["E3"], self.grid["E3"] + UP*0.4, color="#0000FF", stroke_width=10),
            Line(self.grid["E4"], self.grid["E4"] + UP*0.2, color="#0000FF", stroke_width=10)
        )
        
        self.play(FadeIn(ripple1_set), spike1.animate.set_stroke(opacity=0.3))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Second Jump
        jump_path2 = ArcBetweenPoints(start=self.grid["E2"], end=self.grid["E3"], angle=-PI/2)
        self.play(MoveAlongPath(frog, jump_path2), run_time=0.8)
        spike2 = Line(self.grid["E3"], self.grid["E3"] + UP*0.8, color=WHITE, stroke_width=6)
        ripple2_set = VGroup(
            Line(self.grid["E3"], self.grid["E3"] + UP*0.6, color="#0000FF", stroke_width=10),
            Line(self.grid["E4"], self.grid["E4"] + UP*0.4, color="#0000FF", stroke_width=10),
            Line(self.grid["E5"], self.grid["E5"] + UP*0.2, color="#0000FF", stroke_width=10)
        )
        self.play(Create(spike2), FadeIn(ripple2_set))
        
        # Third Jump
        jump_path3 = ArcBetweenPoints(start=self.grid["E3"], end=self.grid["E4"], angle=-PI/2)
        self.play(MoveAlongPath(frog, jump_path3), run_time=0.8)
        spike3 = Line(self.grid["E4"], self.grid["E4"] + UP*0.8, color=WHITE, stroke_width=6)
        ripple3_set = VGroup(
            Line(self.grid["E4"], self.grid["E4"] + UP*0.6, color="#0000FF", stroke_width=10),
            Line(self.grid["E5"], self.grid["E5"] + UP*0.4, color="#0000FF", stroke_width=10),
            Line(self.grid["E6"], self.grid["E6"] + UP*0.2, color="#0000FF", stroke_width=10)
        )
        self.play(Create(spike3), FadeIn(ripple3_set))
        self.wait(1)

        # Calculation of sum (simplified visual representation)
        # We replace the individual bars with a green profile
        # Heights at E2: 0.6
        # Heights at E3: 0.4 + 0.6 = 1.0
        # Heights at E4: 0.2 + 0.4 + 0.6 = 1.2
        # Heights at E5: 0.2 + 0.4 = 0.6
        # Heights at E6: 0.2
        
        green_bars = VGroup(
            Line(self.grid["E2"], self.grid["E2"] + UP*0.6, color="#00FF00", stroke_width=12),
            Line(self.grid["E3"], self.grid["E3"] + UP*1.0, color="#00FF00", stroke_width=12),
            Line(self.grid["E4"], self.grid["E4"] + UP*1.2, color="#00FF00", stroke_width=12),
            Line(self.grid["E5"], self.grid["E5"] + UP*0.6, color="#00FF00", stroke_width=12),
            Line(self.grid["E6"], self.grid["E6"] + UP*0.2, color="#00FF00", stroke_width=12)
        )

        all_elements = VGroup(ripple1_set, ripple2_set, ripple3_set, spike1, spike2, spike3)
        self.play(
            ReplacementTransform(all_elements, green_bars),
            frog.animate.fade(0.5)
        )
        self.wait(2)

        # Cleanup for next section (optional, depending on flow)
        self.play(FadeOut(green_bars), FadeOut(frog), FadeOut(pond_line), self.lecture[4].animate.set_color(WHITE))
        self.wait(1)

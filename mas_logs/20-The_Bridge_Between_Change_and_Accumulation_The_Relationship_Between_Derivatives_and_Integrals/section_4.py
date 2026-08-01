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
        # Setup the layout with title and lecture lines
        lecture_lines = [
            'Calculus uses a "Magic Machine" to connect functions.',
            'First, we feed a position function into the machine.',
            'The differentiation operator calculates the rate of change.',
            'Out comes the velocity function, describing the motion.',
            'Integrating this velocity returns us to the original position.'
        ]
        self.setup_layout("The Core Connection: The Fundamental Theorem of Calculus", lecture_lines)

        # Colors
        COLOR_DIFF = "#F91717" # Red
        COLOR_INT = "#58C4DD"  # Blue
        COLOR_MACHINE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Step 1: Create a white box labeling 'The Magic Machine'
        machine_rect = Rectangle(width=3.8, height=2.2, color=COLOR_MACHINE)
        machine_label = Text("The Magic Machine", font_size=20, color=COLOR_MACHINE)
        # Relative internal offset for label
        machine_label.shift(UP * 0.7)
        machine_group = VGroup(machine_rect, machine_label)
        
        # Issue 44: Positioning the machine away from the left boundary
        self.place_in_area(machine_group, 'B3', 'C6', scale_factor=1.0)
        
        self.play(Create(machine_rect), Write(machine_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        # Step 2: Animate symbol F(x) moving into the left side of the box
        f_cap_x = Text("F(x)", color=WHITE)
        # Issue 43: Move to B2 and scale 0.8 to avoid margin obstruction
        self.place_at_grid(f_cap_x, 'B2', scale_factor=0.8)
        
        self.play(FadeIn(f_cap_x))
        # Move into the machine (target center roughly at B4/C4 area)
        self.play(f_cap_x.animate.move_to(self.grid['B4']))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_DIFF))
        
        # Step 3: Inside the box, flash a red d/dx operator
        diff_op = Text("d/dx", color=COLOR_DIFF)
        # Issue 45: Move to C4 and scale 0.8 to fit internal space
        self.place_at_grid(diff_op, 'C4', scale_factor=0.8)
        
        self.play(Flash(diff_op, color=COLOR_DIFF, flash_radius=0.5))
        self.play(FadeIn(diff_op))
        
        # Transform F(x) to f(x) to represent differentiation
        f_small_x = Text("f(x)", color=WHITE)
        f_small_x.move_to(f_cap_x.get_center())
        
        self.play(
            ReplacementTransform(f_cap_x, f_small_x),
            FadeOut(diff_op, shift=DOWN*0.2)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(WHITE))
        
        # Step 4: Animate symbol f(x) moving out of the right side of the box
        exit_pos = self.grid['B6']
        self.play(f_small_x.animate.move_to(exit_pos))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_INT))
        
        # Step 5: Show f(x) entering an integral chamber return to F(x) + C
        int_rect = Rectangle(width=3.5, height=1.8, color=COLOR_INT)
        int_sym = Text("∫", color=COLOR_INT, font_size=50)
        int_chamber = VGroup(int_rect, int_sym)
        # Place integral chamber in lower right area
        self.place_in_area(int_chamber, 'E3', 'F6', scale_factor=1.0)
        
        self.play(Create(int_rect), Write(int_sym))
        
        # Move f(x) from exit of first machine into the integral chamber
        self.play(f_small_x.animate.move_to(self.grid['E6']))
        self.play(f_small_x.animate.move_to(self.grid['E5']))
        
        # Final result F(x) + C
        result_text = Text("F(x) + C", color=COLOR_INT)
        # Position result_text at the same grid point as entrance to finalize transform
        result_text.scale(0.8).move_to(self.grid['E5'])
        
        self.play(ReplacementTransform(f_small_x, result_text))
        self.play(Indicate(result_text, color=COLOR_INT))
        self.wait(2)

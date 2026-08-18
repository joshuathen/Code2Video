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

class Section6IterationScene(TeachingScene):
    def construct(self):
        title = "The Training Loop: Practice Makes Perfect"
        lines = [
            "Pixel repeats this process thousands of times.",
            "With every example, the knobs become more precise.",
            "Soon, Pixel identifies objects with high accuracy."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_HIGHLIGHT = "#FFFF00"
        COLOR_ARROW = "#FFFFFF"
        COLOR_SUCCESS = "#00FF00"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        
        # Define cycle labels
        forward_lbl = Text("Forward", font_size=24, color=WHITE)
        loss_lbl = Text("Loss", font_size=24, color=WHITE)
        backprop_lbl = Text("Backprop", font_size=24, color=WHITE)
        update_lbl = Text("Update", font_size=24, color=WHITE)

        self.place_at_grid(forward_lbl, "A3")
        self.place_at_grid(loss_lbl, "C6")
        self.place_at_grid(backprop_lbl, "F3")
        self.place_at_grid(update_lbl, "C1")

        # Create curved arrows for the cycle
        arrow1 = CurvedArrow(forward_lbl.get_right(), loss_lbl.get_top(), color=COLOR_ARROW)
        arrow2 = CurvedArrow(loss_lbl.get_bottom(), backprop_lbl.get_right(), color=COLOR_ARROW)
        arrow3 = CurvedArrow(backprop_lbl.get_left(), update_lbl.get_bottom(), color=COLOR_ARROW)
        arrow4 = CurvedArrow(update_lbl.get_top(), forward_lbl.get_left(), color=COLOR_ARROW)

        cycle_group = VGroup(forward_lbl, loss_lbl, backprop_lbl, update_lbl, arrow1, arrow2, arrow3, arrow4)
        
        self.play(Create(cycle_group))
        
        # Rotate a highlight indicator around the cycle to show "repeats"
        dot = Dot(color=COLOR_HIGHLIGHT).move_to(forward_lbl.get_center())
        self.play(Indicate(forward_lbl))
        self.play(MoveAlongPath(dot, arrow1), run_time=0.5)
        self.play(Indicate(loss_lbl))
        self.play(MoveAlongPath(dot, arrow2), run_time=0.5)
        self.play(Indicate(backprop_lbl))
        self.play(MoveAlongPath(dot, arrow3), run_time=0.5)
        self.play(Indicate(update_lbl))
        self.play(MoveAlongPath(dot, arrow4), run_time=0.5)
        self.remove(dot)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_HIGHLIGHT)
        )

        # Represent "Pixel" in the center
        pixel_body = RoundedRectangle(corner_radius=0.1, height=1.5, width=1.5, color=WHITE)
        self.place_in_area(pixel_body, "C3", "D4")
        pixel_label = Text("Pixel", font_size=20).next_to(pixel_body, DOWN, buff=0.1)
        
        # "Knobs" on Pixel
        knob1 = Circle(radius=0.1, color=WHITE).move_to(pixel_body.get_center() + LEFT*0.3 + UP*0.3)
        knob2 = Circle(radius=0.1, color=WHITE).move_to(pixel_body.get_center() + RIGHT*0.3 + UP*0.3)
        knob3 = Circle(radius=0.1, color=WHITE).move_to(pixel_body.get_center() + LEFT*0.3 + DOWN*0.3)
        knob4 = Circle(radius=0.1, color=WHITE).move_to(pixel_body.get_center() + RIGHT*0.3 + DOWN*0.3)
        
        knobs = VGroup(knob1, knob2, knob3, knob4)
        
        # Animation for flashing examples
        example_labels = ["DOG", "TOASTER", "DOG", "CAR", "TOASTER", "BIRD"]
        current_example = Text("", font_size=36, color=WHITE)
        self.place_at_grid(current_example, "B4")

        self.play(FadeIn(pixel_body, pixel_label, knobs), FadeOut(cycle_group))
        
        # Twitching and flashing
        for label_text in example_labels:
            new_text = Text(label_text, font_size=36, color=WHITE)
            self.place_at_grid(new_text, "B4")
            
            # Knobs "twitch" - slight random offsets
            twitch_anims = [k.animate.shift(np.random.uniform(-0.05, 0.05, 3)) for k in knobs]
            
            self.play(
                Transform(current_example, new_text),
                *twitch_anims,
                run_time=0.3
            )
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_HIGHLIGHT)
        )

        # Final identification
        final_dog_example = Text("DOG", font_size=40, color=WHITE)
        self.place_at_grid(final_dog_example, "B4")
        
        # Stop twitching (return to center) and show success
        self.play(
            Transform(current_example, final_dog_example),
            knob1.animate.move_to(pixel_body.get_center() + LEFT*0.3 + UP*0.3),
            knob2.animate.move_to(pixel_body.get_center() + RIGHT*0.3 + UP*0.3),
            knob3.animate.move_to(pixel_body.get_center() + LEFT*0.3 + DOWN*0.3),
            knob4.animate.move_to(pixel_body.get_center() + RIGHT*0.3 + DOWN*0.3),
        )
        
        accuracy_text = Text("99% Accuracy", font_size=24, color=COLOR_SUCCESS)
        self.place_at_grid(accuracy_text, "E4")
        
        self.play(
            current_example.animate.set_color(COLOR_SUCCESS),
            FadeIn(accuracy_text),
            Flash(current_example, color=COLOR_SUCCESS)
        )
        
        self.wait(2)

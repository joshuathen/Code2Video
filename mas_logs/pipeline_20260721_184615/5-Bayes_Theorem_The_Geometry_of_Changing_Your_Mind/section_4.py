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
        # === Setup ===
        title = "The Geometry of Evidence (Filtering)"
        lines = [
            "Evidence has arrived: the sensor just beeped.",
            "We discard all regions where no beep occurred.",
            "Only the \"Beep\" rectangles remain in our view."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_TP = "#00FF00"  # Green: Rain + Beep
        COLOR_FP = "#FF0000"  # Red: No Rain + Beep
        COLOR_DIM = "#555555"
        COLOR_BG = "#333333"
        COLOR_HIGHLIGHT = "#FFFF00"

        # Geometry parameters (normalized to 3.5 units total square)
        W_TOTAL = 3.5
        H_TOTAL = 3.5
        P_RAIN = 0.3
        P_BEEP_RAIN = 0.8
        P_BEEP_NORAIN = 0.2
        
        w_rain = W_TOTAL * P_RAIN
        w_norain = W_TOTAL * (1 - P_RAIN)
        
        h_beep_rain = H_TOTAL * P_BEEP_RAIN
        h_nobeep_rain = H_TOTAL * (1 - P_BEEP_RAIN)
        
        h_beep_norain = H_TOTAL * P_BEEP_NORAIN
        h_nobeep_norain = H_TOTAL * (1 - P_BEEP_NORAIN)

        # Create Rectangles
        # TP: Rain + Beep
        rect_tp = Rectangle(width=w_rain, height=h_beep_rain, fill_color=COLOR_TP, fill_opacity=0.8, stroke_width=2)
        # FN: Rain + No Beep
        rect_fn = Rectangle(width=w_rain, height=h_nobeep_rain, fill_color=COLOR_BG, fill_opacity=0.3, stroke_width=2)
        # FP: No Rain + Beep
        rect_fp = Rectangle(width=w_norain, height=h_beep_norain, fill_color=COLOR_FP, fill_opacity=0.8, stroke_width=2)
        # TN: No Rain + No Beep
        rect_tn = Rectangle(width=w_norain, height=h_nobeep_norain, fill_color=COLOR_BG, fill_opacity=0.3, stroke_width=2)

        # Alignment logic (forming the full square)
        # Column 1 (Rain)
        rect_tp.move_to(ORIGIN) 
        rect_fn.next_to(rect_tp, DOWN, buff=0)
        
        # Column 2 (No Rain)
        rect_fp.align_to(rect_tp, UP).next_to(rect_tp, RIGHT, buff=0)
        rect_tn.next_to(rect_fp, DOWN, buff=0)

        # Create the full square group
        full_square = VGroup(rect_tp, rect_fn, rect_fp, rect_tn)
        # Fix for Issue 33: Move full_square to B3-E6
        self.place_in_area(full_square, "B3", "E6", scale_factor=1.0)

        # Initial Labels
        label_rain = Text("Rain", font_size=18).scale(0.8)
        label_norain = Text("No Rain", font_size=18).scale(0.8)
        
        # Position labels above the columns
        label_rain.next_to(rect_tp, UP, buff=0.2)
        label_norain.next_to(rect_fp, UP, buff=0.2)

        self.add(full_square, label_rain, label_norain)
        self.wait(1.5)

        # === Animation for Lecture Line 1 ===
        # "Evidence has arrived: the sensor just beeped."
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        
        # Asset: sensor.svg (Issue 24)
        sensor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg")
        self.place_at_grid(sensor_icon, "A3", scale_factor=0.5)

        # Dim parts except intersections (TP and FP represent 'Beep')
        self.play(
            FadeIn(sensor_icon),
            rect_fn.animate.set_fill(COLOR_DIM, opacity=0.4).set_stroke(color=COLOR_DIM),
            rect_tn.animate.set_fill(COLOR_DIM, opacity=0.4).set_stroke(color=COLOR_DIM),
            Indicate(rect_tp, color=COLOR_TP),
            Indicate(rect_fp, color=COLOR_FP),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # "We discard all regions where no beep occurred."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT)
        
        # Remove dimmed areas and labels
        self.play(
            FadeOut(rect_fn),
            FadeOut(rect_tn),
            FadeOut(label_rain),
            FadeOut(label_norain),
            FadeOut(sensor_icon),
            run_time=1.5
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # "Only the \"Beep\" rectangles remain in our view."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        
        # Move remaining rectangles to center. 
        beep_group = VGroup(rect_tp, rect_fp)
        
        # Create descriptive labels for the remaining areas
        text_tp = Text("True Positive", font_size=18, color=COLOR_TP)
        text_fp = Text("False Positive", font_size=18, color=COLOR_FP)
        
        # Centering the remaining evidence
        target_center_pos = self.grid["C4"] # Shifted for balance within the grid

        # Fix for Issue 34 and 35: Positioning the labels to avoid overlap
        self.play(
            beep_group.animate.move_to(target_center_pos),
            run_time=2
        )
        
        self.place_in_area(text_tp, "E3", "F3", scale_factor=0.6)
        self.place_in_area(text_fp, "E4", "F5", scale_factor=0.6)

        self.play(
            FadeIn(text_tp),
            FadeIn(text_fp),
            run_time=1.5
        )
        
        # Final emphasis
        self.play(
            Indicate(text_tp, color=COLOR_TP),
            Indicate(text_fp, color=COLOR_FP),
            run_time=1.5
        )
        self.wait(2)
        
        # Cleanup
        self.lecture[2].set_color(WHITE)
        self.wait(2)

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
        # Setup layout with specified title and lecture lines
        title_text = "Summary: Predictability in Unpredictability"
        lecture_lines = [
            "These constants help us model flight and weather patterns.",
            "Universal scaling connects diverse systems through shared math.",
            "Statistical predictability emerges from the heart of chaos.",
            "Mathematics simplifies complex turbulence into solvable curves.",
            "We find profound order within the apparent chaos."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_PLANE_STORM = "#FFFFFF"
        COLOR_LAW_LINE = "#83C167"
        COLOR_PREDICT = "#F8E71C"
        COLOR_MATH_CURVE = "#FFFFFF"
        COLOR_FINAL = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_PLANE_STORM))
        
        # Create Airplane Icon (Simplified)
        plane_body = Ellipse(width=1.2, height=0.3, color=COLOR_PLANE_STORM)
        plane_wing = Triangle(color=COLOR_PLANE_STORM).scale(0.3).rotate(PI)
        plane_icon = VGroup(plane_body, plane_wing)
        # Issue 44: Increased scale to 0.8
        self.place_at_grid(plane_icon, "B2", scale_factor=0.8)
        
        # Create Storm Cloud Icon (Simplified)
        cloud_circles = [Circle(radius=0.4, color=COLOR_PLANE_STORM, fill_opacity=0.3).shift(pos) 
                         for pos in [LEFT*0.3, RIGHT*0.3, UP*0.2]]
        storm_icon = VGroup(*cloud_circles)
        # Issue 44: Increased scale to 0.8
        self.place_at_grid(storm_icon, "E5", scale_factor=0.8)
        
        self.play(FadeIn(plane_icon), FadeIn(storm_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_LAW_LINE))
        
        # Power law line connects icons
        law_line = Line(
            start=self.grid["B2"], 
            end=self.grid["E5"], 
            color=COLOR_LAW_LINE, 
            stroke_width=6
        )
        self.play(Create(law_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_PREDICT))
        
        # Text "STATISTICAL PREDICTABILITY"
        predict_text = Text("STATISTICAL PREDICTABILITY", font_size=20, color=COLOR_PREDICT)
        # Issue 42: Changed position to area B4-C6 and scale to 0.7 to avoid overlap
        self.place_in_area(predict_text, 'B4', 'C6', scale_factor=0.7)
        
        self.play(Write(predict_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_MATH_CURVE))
        
        # Chaotic background - random lines to represent chaos
        chaos_lines = VGroup()
        for _ in range(15):
            start_p = self.grid["A1"] + np.random.uniform(-0.5, 0.5, 3)
            end_p = self.grid["F6"] + np.random.uniform(-0.5, 0.5, 3)
            chaos_lines.add(Line(start_p, end_p, color=BLUE, stroke_opacity=0.2))
        
        self.add_foreground_mobjects(plane_icon, storm_icon, law_line, predict_text)
        self.play(FadeIn(chaos_lines))
        
        # Simplify background: Chaotic lines fade as a clean curve appears
        clean_curve = FunctionGraph(
            lambda x: 0.5 * np.sin(x * 1.5) - 0.2 * x,
            x_range=[-2.5, 2.5],
            color=COLOR_MATH_CURVE
        )
        self.place_in_area(clean_curve, "B1", "E6", scale_factor=1.0)
        
        self.play(
            FadeOut(chaos_lines),
            Create(clean_curve),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_FINAL))
        
        # Final text "Order within chaos" glows
        final_text = Text("Order within chaos", font_size=32, color=COLOR_FINAL)
        # Issue 43: Moved to F1-F4 area and scaled to 0.8 to resolve crowding
        self.place_in_area(final_text, 'F1', 'F4', scale_factor=0.8)
        
        self.play(Write(final_text))
        self.play(Indicate(final_text, color=COLOR_FINAL, scale_factor=1.1))
        self.wait(2)

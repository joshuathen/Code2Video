from manim import *

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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Geometry of a Vector", [
            "Visually, we represent a vector as an arrow.",
            "Every vector has a specific length, called magnitude.",
            "Every vector also points in a specific direction.",
            "You can slide a vector anywhere on the grid.",
            "It remains the same if magnitude and direction persist."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Visually, we represent a vector as an arrow.
        vector_color = "#00FFFF"
        vector = Arrow(self.grid["D3"], self.grid["B5"], buff=0, color=vector_color, stroke_width=8)
        vector_label = Text("Vector", font_size=24, color=vector_color)
        # Fix for Issue 23: move to C6, scale 0.8
        self.place_at_grid(vector_label, "C6", scale_factor=0.8)
        
        self.lecture[0].set_color(vector_color)
        self.play(Create(vector), Write(vector_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Every vector has a specific length, called magnitude.
        mag_color = "#FFFF00"
        brace = BraceBetweenPoints(self.grid["D3"], self.grid["B5"], color=mag_color)
        mag_label = Text("Magnitude", font_size=20, color=mag_color).next_to(brace, LEFT, buff=0.1)
        
        self.lecture[1].set_color(mag_color)
        self.play(Create(brace), Write(mag_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Every vector also points in a specific direction.
        dir_color = "#FFA500"
        # Reference line and arc to show direction/angle
        line_ref = DashedLine(self.grid["D3"], self.grid["D3"] + RIGHT * 1.5, color=WHITE, stroke_opacity=0.5)
        angle_arc = Arc(radius=0.5, start_angle=0, angle=PI/4, arc_center=self.grid["D3"], color=dir_color)
        dir_label = Text("Direction", font_size=20, color=dir_color)
        # Fix for Issue 24: move to F3, scale 0.8
        self.place_at_grid(dir_label, "F3", scale_factor=0.8)
        
        self.lecture[2].set_color(dir_color)
        self.play(Create(line_ref), Create(angle_arc), Write(dir_label), Flash(vector.get_end(), color=dir_color))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # You can slide a vector anywhere on the grid.
        slide_color = "#00FF00"
        self.lecture[3].set_color(slide_color)
        
        # Group the vector and its properties to move them together
        vector_group = VGroup(vector, vector_label, brace, mag_label, line_ref, angle_arc, dir_label)
        
        # Slide to B1 (top-left area)
        shift_vector = self.grid["B1"] - self.grid["D3"]
        self.play(vector_group.animate.shift(shift_vector), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # It remains the same if magnitude and direction persist.
        persist_color = "#FF00FF"
        self.lecture[4].set_color(persist_color)
        
        # Slide to E4 (bottom-right area)
        shift_vector2 = self.grid["E4"] - self.grid["B1"]
        self.play(vector_group.animate.shift(shift_vector2), run_time=2)
        
        # Pulse the arrow to show it is unchanged
        self.play(
            vector.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=0.5
        )
        self.wait(2)

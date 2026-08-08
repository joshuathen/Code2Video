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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        title_text = "The Big Idea: Beyond Numbers"
        lecture_lines = [
            "Meet the vector, a quantity with size and direction.",
            "Unlike scalars, vectors tell us where we are going.",
            "We represent vectors as arrows pointing through space."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Meet the vector, a quantity with size and direction.
        # Display the scalar value "5m" in white (#FFFFFF) at the center of the right-side visual area.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        scalar_val = Text("5m", color=WHITE)
        # Fix as per Issue 16: Move scalar_val to B2-C3 area with scale 1.0 to avoid trajectory overlap
        self.place_in_area(scalar_val, "B2", "C3", scale_factor=1.0)
        
        self.play(Write(scalar_val))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Unlike scalars, vectors tell us where we are going.
        # Transform "5m" into a yellow (#FFFF00) arrow pointing North-East.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Define arrow from D3 to B5 (North-East direction in the grid)
        start_pt = self.grid["D3"]
        end_pt = self.grid["B5"]
        vector_arrow = Arrow(start=start_pt, end=end_pt, color="#FFFF00", buff=0)
        
        self.play(ReplacementTransform(scalar_val, vector_arrow))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We represent vectors as arrows pointing through space.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # 3.1 Pulsate length (emphasize magnitude)
        self.play(vector_arrow.animate.scale(1.2), run_time=0.4, rate_func=there_and_back)
        self.play(vector_arrow.animate.scale(0.8), run_time=0.4, rate_func=there_and_back)
        
        # 3.2 Flash the arrowhead green (#00FF00) three times to emphasize direction.
        tip = vector_arrow.tip
        for _ in range(3):
            self.play(tip.animate.set_color("#00FF00"), run_time=0.2)
            self.play(tip.animate.set_color("#FFFF00"), run_time=0.2)
            
        # 3.3 Move the entire arrow across the grid to illustrate spatial displacement.
        # Displacement: Shift by 1 unit right and 1 unit down (D3->E4, B5->C6)
        # Using ValueTracker for displacement
        shift_tracker = ValueTracker(0)
        initial_pos = vector_arrow.get_center()
        
        def arrow_updater(m):
            s = shift_tracker.get_value()
            m.move_to(initial_pos + s * (RIGHT + DOWN))
            
        vector_arrow.add_updater(arrow_updater)
        self.play(shift_tracker.animate.set_value(1), run_time=1.5)
        vector_arrow.remove_updater(arrow_updater)
        
        self.wait(1)
        
        # Cleanup
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)

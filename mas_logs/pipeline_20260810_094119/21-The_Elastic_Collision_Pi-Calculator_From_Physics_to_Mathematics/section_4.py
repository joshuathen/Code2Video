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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualization and Scaling", [
            "Ratio 100^0 gives 3 collisions.", 
            "Ratio 100^1 gives 31 collisions.", 
            "Ratio 100^2 gives 314 collisions."
        ])
        
        # Setup counter
        counter_val = ValueTracker(0)
        # Creating DecimalNumber outside always_redraw per rule 11
        counter_obj = DecimalNumber(
            0, 
            num_decimal_places=0, 
            color=YELLOW
        )
        
        # Use updater instead of always_redraw for performance
        counter_obj.add_updater(lambda m: m.set_value(counter_val.get_value()))
        
        # Fix 45: place_at_grid(counter, 'C1', scale_factor=0.8)
        self.place_at_grid(counter_obj, 'C1', scale_factor=0.8)
        self.add(counter_obj)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(counter_val.animate.set_value(3), run_time=1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.play(counter_val.animate.set_value(31), run_time=1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        self.play(counter_val.animate.set_value(314), run_time=2)
        
        # Highlight with Pi digits
        pi_digits = Text("3.14", color="#00FF00", font_size=48)
        # Fix 46: self.place_in_area(pi_digits, 'E1', 'F2', scale_factor=0.9)
        self.place_in_area(pi_digits, 'E1', 'F2', scale_factor=0.9)
        
        # Asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg]
        sphere_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg").scale(0.4)
        
        # Fix 47: group them and place
        summary_group = VGroup(pi_digits, sphere_icon).arrange(RIGHT)
        self.place_in_area(summary_group, 'A4', 'F6', scale_factor=0.7)
        
        self.play(FadeIn(summary_group))
        self.wait(2)

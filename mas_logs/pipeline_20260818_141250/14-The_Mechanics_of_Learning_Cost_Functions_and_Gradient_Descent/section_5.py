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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Iterative loops refine our model's weights.", "Predict, calculate cost, and adjust slope.", "This cycle is machine learning."]
        self.setup_layout("Summary and Synthesis", lecture_lines)
        
        # Elements
        loop_title = Text("The Iterative Learning Loop", font_size=32, color=WHITE)
        self.place_at_grid(loop_title, 'A4', scale_factor=0.9) # Fix for Issue 35
        
        # Flow Chart Elements
        step1 = Text("Predict").scale(0.7)
        step2 = Text("Cost").scale(0.7)
        step3 = Text("Gradient").scale(0.7)
        step4 = Text("Adjust").scale(0.7)
        
        loop_group = VGroup(step1, step2, step3, step4)
        
        # Correct placement
        self.place_in_area(loop_group, 'B2', 'D5', scale_factor=0.9) # Fix for Issue 33
        
        # Need to re-position individual items because place_in_area transformed the group
        # but they were placed at grid before. Let's arrange them properly inside the area.
        step1.move_to(self.grid['B3'])
        step2.move_to(self.grid['B5'])
        step3.move_to(self.grid['D5'])
        step4.move_to(self.grid['D3'])
        
        arrows = VGroup(
            Arrow(step1.get_right(), step2.get_left()),
            Arrow(step2.get_bottom(), step3.get_top()),
            Arrow(step3.get_left(), step4.get_right()),
            Arrow(step4.get_top(), step1.get_bottom())
        )
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/valley.svg
        valley = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/valley.svg").scale(0.5)
        self.place_at_grid(valley, 'E5')

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(loop_title))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(VGroup(step1, step2, step3, step4, arrows)))
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Highlight loop
        for obj in [step1, step2, step3, step4]:
            self.play(obj.animate.set_color("#00FF00"), run_time=0.5)
            
        # Highlight valley
        self.play(valley.animate.set_color("#00FF00"), run_time=1.0)
            
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        final_text = Text("Learning Complete", font_size=40, color=WHITE)
        self.place_at_grid(final_text, 'E4', scale_factor=1.0) # Fix for Issue 34
        self.play(Write(final_text))
        self.wait(2)

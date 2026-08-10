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
        self.setup_layout("Summary and Connection: The Universal Equation", 
                          ["Three operations, one mathematical story.", 
                           "Shift your focus to find the unknown.", 
                           "Powers, roots, logs: linked forever."])
        
        # Define equations
        eq_pow = MathTex("2^3 = 8", color=YELLOW)
        eq_root = MathTex(r"\sqrt[3]{8} = 2", color="#00FFFF")
        eq_log = MathTex(r"\log_2(8) = 3", color=PINK)
        
        # Central icon
        # Placeholder for [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        # In a real environment, we'd use SVGMobject or similar if the path exists.
        # Here, we use a simple Dot as a placeholder for the asset if it's missing or effectively empty.
        central_icon = Dot(color=WHITE, radius=0.2)
        
        # Grouping for layout
        root_log_group = VGroup(eq_root, eq_log).arrange(RIGHT, buff=0.5)
        combined_eq_group = VGroup(eq_pow, root_log_group, central_icon).arrange(DOWN, buff=0.8)
        
        # Layout according to requirements
        self.place_at_grid(eq_pow, 'B3', scale_factor=1.2)
        self.place_in_area(root_log_group, 'D2', 'D5', scale_factor=1.0)
        self.place_in_area(combined_eq_group, 'C2', 'E5', scale_factor=1.1)
        
        cycle_arrow = CurvedArrow(eq_pow.get_bottom(), eq_root.get_right(), radius=1.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(eq_pow), FadeIn(eq_root), FadeIn(eq_log), FadeIn(central_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.play(Indicate(eq_pow), Indicate(eq_root), Indicate(eq_log))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(PINK))
        self.play(Create(cycle_arrow))
        self.wait(2)

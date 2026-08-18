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
        self.setup_layout("Conclusion: The Lesson of Rigor", [
            "Observation is not mathematical proof.", 
            "Hidden constraints change the result.", 
            "Always rely on the formula."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Summarize the pizza slices count clearly (#FFFFFF)
        # Using SVGMobject for asset integration
        pizza_1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pizza.svg", color="#FFFFFF")
        text1 = Text("Regions = 2^(n-1)", color="#FFFFFF")
        group1 = VGroup(pizza_1, text1).arrange(DOWN, buff=0.2)
        self.place_at_grid(group1, "B1", scale_factor=0.6)
        self.lecture[0].set_color("#FFFFFF")
        self.play(FadeIn(group1))

        # === Animation for Lecture Line 2 ===
        # Emphasize the danger of assuming simple linear patterns
        # Use a warning sign graphic concept
        rect = RoundedRectangle(corner_radius=0.2, height=1.5, width=3, color=RED)
        warning = Text("DANGER: Linear Trap!", font_size=24, color=RED)
        vgroup = VGroup(rect, warning)
        self.place_in_area(vgroup, "D3", "F6", scale_factor=0.7)
        self.lecture[1].set_color("#FF6347")
        self.play(Create(rect), Write(warning))

        # === Animation for Lecture Line 3 ===
        # Display a final visual of the complex slice distribution (#FFD700)
        pizza_2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pizza.svg", color="#FFD700")
        self.place_at_grid(pizza_2, "A4", scale_factor=0.7)
        self.lecture[2].set_color("#FFD700")
        self.play(FadeIn(pizza_2), run_time=2)
        self.wait(2)

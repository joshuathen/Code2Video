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
        self.setup_layout("The Grand Reveal", [
            "Collisions map to digits of pi.", 
            "Physics creates transcendental numbers.", 
            "Geometry reveals the hidden sequence."
        ])
        
        # Assets
        billiard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/billiard.svg")
        
        # Objects
        equation = MathTex(r"N \approx \pi \cdot 10^n", color=WHITE)
        geometry_viz = Circle(radius=1.5, color="#00FF00", fill_opacity=0.3)
        billiard_icon.set_color(WHITE)
        
        # Combined group
        combined_elements = VGroup(equation, geometry_viz, billiard_icon).arrange(DOWN)
        
        # === Animation for Lecture Line 1 ===
        # Position equation at B5 per Critic
        self.place_at_grid(equation, "B5", scale_factor=0.8)
        self.play(FadeIn(equation), FadeIn(billiard_icon.scale(0.5).next_to(equation, UP)))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Position geometry at D5 per Critic
        self.place_at_grid(geometry_viz, "D5", scale_factor=0.7)
        self.play(Create(geometry_viz))
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Use place_in_area for combined_elements as requested
        self.place_in_area(combined_elements, "B4", "D6", scale_factor=0.75)
        self.play(Indicate(combined_elements, color="#FFFF00"))
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(2)

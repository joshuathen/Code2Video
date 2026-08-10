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
        self.setup_layout("Prerequisite Physics: Conservation Laws", 
                          ["Collisions conserve both momentum and energy.", 
                           "Velocity updates define a system of equations.", 
                           "Phase space plots show circular paths."])
        
        # Assets
        pendulum_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pendulum.svg")
        billiard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/billiard.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#90EE90")
        k_text = Text("K", color="#90EE90").scale(0.8)
        p_text = Text("P", color="#90EE90").scale(0.8)
        energy_momentum = VGroup(k_text, p_text).arrange(RIGHT, buff=1.0)
        self.place_in_area(energy_momentum, 'B2', 'B5', scale_factor=1.0)
        self.play(FadeIn(energy_momentum))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFA500")
        v_vector = Arrow(start=ORIGIN, end=RIGHT*1.5, color="#FFA500")
        v_label = Text("v", color="#FFA500").next_to(v_vector, UP)
        pendulum_icon.set_color("#FFA500")
        
        physics_animation_group = VGroup(v_vector, v_label, pendulum_icon)
        self.place_at_grid(v_vector, 'D3', scale_factor=0.8) # Fix from issue 22/35
        self.place_at_grid(pendulum_icon, 'D5', scale_factor=0.8) # Fix from issue 22/35
        
        # Using place_in_area as per issue 23
        self.place_in_area(physics_animation_group, 'C2', 'E5', scale_factor=0.9)
        
        self.play(FadeIn(physics_animation_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#32CD32")
        billiard_icon.set_color("#32CD32")
        resultant_vector = VGroup(billiard_icon, Circle(radius=0.5, color="#32CD32"))
        self.place_at_grid(resultant_vector, 'E4', scale_factor=0.8) # Fix from issue 24
        
        self.play(FadeIn(resultant_vector))
        self.wait(2)

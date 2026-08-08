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
        self.setup_layout("Synthesis: The Weighted Sum", [
            "Output is a weighted sum of values.", 
            "Relevant words dominate the final context.", 
            "Unimportant words are effectively ignored.", 
            "The context vector captures semantic meaning.", 
            "Now 'it' refers correctly to 'animal'."
        ])
        
        # Load asset
        animal_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/animal.svg")
        self.place_at_grid(animal_icon, "A5", scale_factor=0.3)
        self.add(animal_icon)
        
        # Define colored Value vectors
        v_colors = [BLUE, GREEN, YELLOW, RED, PURPLE]
        v_vectors = VGroup(*[Arrow(ORIGIN, UP*0.8, color=v_colors[i], buff=0) for i in range(5)])
        
        # Position vectors
        for i, v in enumerate(v_vectors):
            self.place_at_grid(v, f"C{i+1}", scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(v_vectors))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        context_vector = Arrow(ORIGIN, UP*1.2, color=WHITE, buff=0)
        # Using recommendation from issue 31
        self.place_at_grid(context_vector, "C4", scale_factor=0.75)
        
        transformations = [
            v_vectors[i].animate.move_to(context_vector.get_start()) 
            for i in range(5)
        ]
        
        # Asset integration
        animal_result = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/animal.svg")
        self.place_at_grid(animal_result, "C4", scale_factor=0.5)
        animal_result.set_opacity(0)
        
        self.play(*transformations, FadeIn(context_vector))
        self.play(FadeIn(animal_result))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(FadeOut(v_vectors[2]), FadeOut(v_vectors[3]))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(RED))
        self.play(context_vector.animate.set_color(GOLD))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE))
        self.wait(2)

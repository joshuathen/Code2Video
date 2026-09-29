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
        self.setup_layout("Prerequisites: The Vector Space Concept", 
                          ["Embeddings map words to coordinate points.", 
                           "Machines calculate distance between these vectors.", 
                           "Conceptually similar words are placed close together."])
        
        # === Animation for Lecture Line 1 ===
        # Create grid and two icons representing "Cat" and "Dog"
        cat_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg")
        dog_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg")
        
        self.place_at_grid(cat_icon, 'C2', scale_factor=0.5)
        self.place_at_grid(dog_icon, 'D4', scale_factor=0.5)
        
        cat_label = Text("Cat", font_size=18, color=WHITE).next_to(cat_icon, UP, buff=0.1)
        dog_label = Text("Dog", font_size=18, color=WHITE).next_to(dog_icon, UP, buff=0.1)
        
        self.play(FadeIn(cat_icon), FadeIn(dog_icon), FadeIn(cat_label), FadeIn(dog_label))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 2 ===
        # Draw a dashed line between the two points
        dist_line = DashedLine(cat_icon.get_center(), dog_icon.get_center(), color="#FF00FF")
        self.play(Create(dist_line))
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        
        # === Animation for Lecture Line 3 ===
        # Move the two points closer together
        new_cat_pos = self.grid['C3']
        new_dog_pos = self.grid['C4']
        
        self.play(
            cat_icon.animate.move_to(new_cat_pos),
            cat_label.animate.move_to(new_cat_pos + UP * 0.5),
            dog_icon.animate.move_to(new_dog_pos),
            dog_label.animate.move_to(new_dog_pos + UP * 0.5),
            UpdateFromFunc(dist_line, lambda m: m.put_start_and_end_on(cat_icon.get_center(), dog_icon.get_center()))
        )
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.wait(1)

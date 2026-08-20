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
        self.setup_layout("Summary and Real-World Application", 
                          ["Vectors simplify movement in physics simulations.", 
                           "Component addition determines object paths.", 
                           "Game characters use these for jumping."])
        
        # Asset Loading
        char_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/character.png"
        char_img = ImageMobject(char_asset)

        # === Animation for Lecture Line 1 ===
        # Fade in a collection of previous vector examples
        v1 = Vector([1, 1], color=WHITE)
        v2 = Vector([-0.5, 1.5], color=WHITE)
        vecs = VGroup(v1, v2).arrange(RIGHT, buff=0.5)
        
        # Placing assets and vectors based on constraints
        # Use columns 4-6 (B004)
        self.place_in_area(vecs, 'A4', 'C6', scale_factor=0.7)
        self.place_at_grid(char_img, 'B3', scale_factor=0.3)
        
        self.play(FadeIn(vecs), FadeIn(char_img))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Animate the vectors interacting in a simple physics simulation
        v_sum = Vector([0.5, 2.5], color="#00BFFF")
        self.place_at_grid(v_sum, 'D5', scale_factor=0.9)
        self.play(vecs.animate.set_opacity(0.3), GrowArrow(v_sum))
        self.lecture[1].set_color("#00BFFF")

        # === Animation for Lecture Line 3 ===
        # Display text 'Vectors govern physical motion' on screen
        motion_text = Text("Vectors govern physical motion", font_size=24, color="#FFD700")
        self.place_in_area(motion_text, 'E2', 'F5', scale_factor=0.7)
        self.play(Write(motion_text))
        self.lecture[2].set_color("#FFD700")
        self.wait(2)

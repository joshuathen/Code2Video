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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Vector Addition: The Tip-to-Tail Method", 
                          ["Add vectors using the tip-to-tail method.", 
                           "Place the second vector on the first's tip.", 
                           "The resultant is the shortcut from start."])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF00FF")
        # u asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg is just a placeholder, using icon if exists or dummy.
        # Assuming SVG loading
        try:
            u_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg").scale(0.2)
        except:
            u_icon = Dot(color=WHITE).scale(0.2)
            
        u = Arrow(start=self.grid["D3"], end=self.grid["C5"], buff=0, color="#FF00FF")
        u_label = MathTex("u", color="#FF00FF").next_to(u.get_center(), UP, buff=0.1)
        u_icon.next_to(u_label, LEFT)
        
        self.play(Create(u), Write(u_label), FadeIn(u_icon))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        v = Arrow(start=u.get_end(), end=self.grid["B6"], buff=0, color="#00FFFF")
        v_label = MathTex("v", color="#00FFFF").next_to(v.get_center(), RIGHT, buff=0.1)
        self.play(Create(v), Write(v_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        w = Arrow(start=u.get_start(), end=v.get_end(), buff=0, color="#FFFF00")
        w_label = MathTex("w = u + v", color="#FFFF00")
        self.place_at_grid(w_label, 'F3', scale_factor=0.7)
        
        # Load second icon
        try:
            w_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg").scale(0.2)
        except:
            w_icon = Dot(color=YELLOW).scale(0.2)
        w_icon.next_to(w_label, RIGHT)
            
        self.play(Create(w), Write(w_label), FadeIn(w_icon))
        self.wait(2)

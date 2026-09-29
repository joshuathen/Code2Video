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
        lecture_lines = ["We compressed information into the board's state.", 
                        "The impossible task is solved through smart encoding.", 
                        "Strategic communication grants the prisoners their freedom."]
        self.setup_layout("Conclusion: The Power of Compressed Communication", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Summarize key points in light cyan (#E0FFFF) text boxes.
        box1 = RoundedRectangle(corner_radius=0.1, color="#E0FFFF", height=1.0, width=3.5)
        text1 = Text("Compressed Info:\nBoard State = Instruction", font_size=18, color="#E0FFFF")
        group1 = VGroup(box1, text1)
        self.place_in_area(group1, 'B4', 'C6', scale_factor=0.8)
        self.play(FadeIn(group1))
        self.play(self.lecture[0].animate.set_color("#E0FFFF"))

        # === Animation for Lecture Line 2 ===
        # Animate a handshake icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/handshake.svg] in light blue (#ADD8E6).
        handshake = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/handshake.svg").set_color("#ADD8E6")
        box2 = RoundedRectangle(corner_radius=0.1, color="#ADD8E6", height=1.0, width=3.5)
        text2 = Text("Smart Encoding\nSolves the Task", font_size=18, color="#ADD8E6")
        group2 = VGroup(box2, text2, handshake)
        group2.arrange(DOWN)
        self.place_in_area(group2, 'D4', 'E6', scale_factor=0.8)
        self.play(FadeIn(group2))
        self.play(self.lecture[1].animate.set_color("#ADD8E6"))

        # === Animation for Lecture Line 3 ===
        # Fade to a closing message in soft white (#FFFFFF).
        final_msg = Text("Prisoners are FREE!", font_size=24, color=WHITE)
        self.place_in_area(final_msg, 'F2', 'F6', scale_factor=0.9)
        self.play(FadeIn(final_msg))
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.wait(2)
